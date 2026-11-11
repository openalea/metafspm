"""
metafspm's DataStructures as encoding3d frames (the cross-platform scene format, https://github.com/openalea/encoding3d),
and back::

    frame = to_frame({"plants": ds, "soil": grid}, components={"plants": [growth, carbon]},
                     mappings={"roots_in_soil": (cross_mapping, "plants", "soil")})
    frame = scene_to_frame(scene)                  every population's and environment model's DataStructure, and the
                                                   scene's mappings
    export_frame(path, ...), export_scene(path, scene)       the same, written to a .glb file
    data_structures = from_frame(frame)            {id: DataStructure}, rebuilt: import_frame(path) from a file
    write_views(path, default_views(scene))        the layers ParaView shows (views.json): the shoots, the roots as
                                                   tubes, the soil grids

What a frame holds:

- an MPGDataStructure: one entity set per MPG scale holding vertices (Plant, Axis, ... SubOrgan, Layer, Cell,
  Compartment; ids are the vids), the Connections as a relation (n_id_a → n_id_b); as columns, every MPG property
  and every variable of the DataStructure at that scale's location; its scalars; the triangles of its geometry
  provider as a mesh, by node and by coarser scale, and a mesh for each component drawing shapes for display only
  (display_triangles(), e.g. GRANAP's sections, its vertices shared by its triangles; display_meshes=False leaves them
  out, SceneRecorder writes them every display_period steps only; display_plants= draws them for some plants only);
  the groups of its Reduction (any, at any location) drawn as instances of their templates (display="instances",
  the default: an instance set per component giving display_instances, its templates and each archetype's recovery
  operator shared parts written once, the reduced system's unknown on the templates' nodes a derived field the reader
  rebuilds for the instances it selects), or whole with every interior as display rows (display="full").
  The MPG's own links (the raw complex and parent of each
  vertex, its raw edge type) are the metafspm_complex, metafspm_parent and metafspm_edge_type columns, and the
  label codes, variable metadata and construction arguments go in the data structure's extras, so that
  from_frame rebuilds the same MPG (vertices, links, properties) and the same variables, bit for bit;
- an ArrayDataStructure: a regular grid (its cells in C order, its faces as a relation); an
  AdaptiveGridDataStructure: an adaptive grid (its leaves' level and index);
- a CrossMapping: its incidence (rows, columns, weights) as a mapping, which from_frame leaves out (the scene
  rebuilds its mappings);
- declarations: the unit (as a UCUM code), kind, role, description, range and references of each variable
  declared by the components given, with a standard name where the vocabulary's metafspm column lists the
  variable's name (its declared unit converting to the standard name's).

Not exchanged: derivation formulas and mask rules (code), random streams, the components' own state.
"""
from dataclasses import fields, is_dataclass
import json
import math
import re
import warnings

import numpy as np

from openalea.metafspm.coupling.declaration import VariableSpec
from openalea.metafspm.data_structure.adaptive_grid import AdaptiveGridDataStructure
from openalea.metafspm.data_structure.arraydict import ArrayDict
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG

PLATFORM = "openalea.metafspm"
EXTRAS = "metafspm"
NO_ID = -1
EDGE_CODES = {1: "/", 2: "<", 3: "+"}               # the MPG's edge types (MPG.topology_arrays)
LINK_COLUMNS = ("metafspm_complex", "metafspm_parent", "metafspm_edge_type")
DISPLAY_ONLY = "metafspm_display_only"             # the display rows' column: a reduced section's interior vertices
DISPLAYS = ("instances", "full", None)             # how display shapes draw a Reduction's groups (to_frame's display)

_KINDS = {"intensive": "intensive", "extensive": "extensive", "massic_concentration": "massic_concentration",
          "volumic_concentration": "volumic_concentration", "descriptor": "descriptor",
          "non_inertial_intensive": "intensive", "non_inertial_extensive": "extensive"}
_ROLES = {"state_variable": "state", "input_variable": "input", "parameter": "parameter",
          "setting": "setting", "forcing_variable": "input"}
_DIMENSIONLESS = {"", "adim", "-", "1", "dimensionless", "adimensional", "unitless"}
_SYMBOLS = {"°C": "Cel", "°": "deg", "day": "d", "days": "d", "yr": "a", "°Cd": "K.d", "°C.d": "K.d"}
_GRAPH_RESERVED = ("id", "label", "complex", "parent", "edge_type", "sibling_order", "level", "index")
_RELATION_RESERVED = ("id", "source", "target", "relation")


def _encoding3d():
    try:
        from openalea import encoding3d
    except ImportError:
        raise ImportError("openalea.encoding3d, a dependency of metafspm, is not installed (in development: clone encoding3d "
                          "next to metafspm, then pip install -e ../encoding3d)") from None
    return encoding3d


# ── Units, standard names, declarations ─────────────────────────────────────

def ucum_unit(unit) -> str:
    """The UCUM code of a metafspm declaration's unit: 'mm3 s-1 MPa-1' → 'mm3.s-1.MPa-1', '°C' → 'Cel', 'adim' →
    '1', 'mol of hexose' → 'mol{hexose}'; a unit with no UCUM code becomes an annotation ('{phyllochron}')."""
    from openalea.encoding3d import ucum
    text = "" if unit is None or (isinstance(unit, float) and math.isnan(unit)) else str(unit).strip()
    if text in _DIMENSIONLESS:
        return "1"
    quotient = re.fullmatch(r"(.+?)\s+per\s+\((.+)\)", text)
    if quotient:                                        # "A per (B)": A / (B)
        numerator, denominator = ucum_unit(quotient.group(1)), ucum_unit(quotient.group(2))
        if not numerator.startswith("{") and not denominator.startswith("{"):
            return f"{numerator}/({denominator})"
    qualifier = ""
    found = re.search(r"\s+of\s+(.+)$", text)
    if found:
        qualifier, text = _annotation(found.group(1)), text[:found.start()]
    factors = []
    for factor in re.split(r"[\s·]+|(?<=[A-Za-z0-9])\.(?=[A-Za-zµ°%])", text.replace("µ", "u")):
        if not factor:
            continue
        parts = re.fullmatch(r"(.*?)(\^|\*\*)?(-?\d+)?", factor)
        symbol, exponent = parts.group(1), parts.group(3) or ""
        factors.append(_SYMBOLS.get(symbol, symbol) + exponent)
    code = ".".join(factors) + qualifier
    if code and ucum.is_valid(code):
        return code
    return _annotation(str(unit))


def _annotation(text: str) -> str:
    """A UCUM annotation of *text*: printable ASCII, no spaces or braces."""
    text = text.strip().replace("µ", "u").replace("°", "deg")
    return "{" + re.sub(r"[^!-z|~]", "_", re.sub(r"[{}]", "", text)) + "}"


def _declarations(components) -> dict:
    """{variable name: declaration metadata} of the variables the *components* declare (the first declaring each)."""
    found = {}
    for component in components or ():
        if not is_dataclass(component):
            continue
        for item in fields(type(component)):
            if "role" in item.metadata or "unit" in item.metadata:
                found.setdefault(item.name, dict(item.metadata))
    return found


RECORDED = "recorded"         # a variables filter's value: what the declarations record (record=; Excluding)


class Excluding:
    """A variables filter's value for a table: every variable but *names* (a filter given as "recorded": every
    variable but those no declaration records)."""

    def __init__(self, names):
        self.names = frozenset(str(name) for name in names)

    def __contains__(self, name) -> bool:
        return name not in self.names

    def __or__(self, other) -> "Excluding":
        return Excluding(self.names - set(other))

    def __eq__(self, other) -> bool:
        return isinstance(other, Excluding) and other.names == self.names

    def __hash__(self) -> int:
        return hash(("Excluding", self.names))

    def __repr__(self) -> str:
        return f"Excluding({sorted(self.names)})"


def unrecorded(ds, components) -> set:
    """The variables of *ds* its frames leave out under a variables filter "recorded": those whose declarations by
    *components* (record=, coupling.component's declarations) do not record them: a declaration's explicit record=
    (True wins), else the DataStructure's own (a grid's geometric descriptors: record=False), else a state variable
    or an input recorded, a parameter or a setting not. A variable nothing declares is recorded."""
    roles, explicit = {}, {}
    for component in components or ():
        if not is_dataclass(component):
            continue
        for item in fields(type(component)):
            role = item.metadata.get("role")
            if role is None:
                continue
            roles.setdefault(item.name, set()).add(role)
            if item.metadata.get("record") is not None:
                explicit[item.name] = explicit.get(item.name, False) or bool(item.metadata["record"])
    meta = ds._variable_meta() if hasattr(ds, "_variable_meta") else {}
    out = set()
    for name in set(roles) | {name for name, info in meta.items() if info.get("record") is not None}:
        if name in explicit:
            recorded = explicit[name]
        elif meta.get(name, {}).get("record") is not None:
            recorded = bool(meta[name]["record"])
        else:
            recorded = bool(roles.get(name, set()) & {"state_variable", "input_variable"})
        if not recorded:
            out.add(name)
    return out


def resolved_variables(variables, ds, components):
    """A variables filter's value for *ds* with "recorded" (as a whole, or as a table's value) resolved into
    Excluding(unrecorded(ds, components))."""
    if variables is None:
        return None
    if isinstance(variables, str) and variables == RECORDED:
        return Excluding(unrecorded(ds, components))
    if isinstance(variables, dict):
        return {table: (Excluding(unrecorded(ds, components)) if isinstance(names, str) and names == RECORDED
                        else names) for table, names in variables.items()}
    return variables


def _variable(name: str, metadata: dict, standard_names: dict, derived: bool = False):
    encoding3d = _encoding3d()
    from openalea.encoding3d import ucum, vocabulary
    unit = ucum_unit(metadata.get("unit"))
    standard_name = standard_names.get(name)
    entry = vocabulary.standard_names().get(standard_name) if standard_name else None
    if entry is None or not ucum.convertible(unit, entry["canonical_unit"]):
        standard_name = None
    description = metadata.get("description") or None
    if metadata.get("unit_comment"):
        description = f"{description or ''} ({metadata['unit_comment']})".strip()
    if unit.startswith("{") and metadata.get("unit"):
        description = f"{description or ''} [unit as declared: {metadata['unit']}]".strip()
    if metadata.get("upscaling") is not None:
        description = f"{description or ''} [upscaling: {metadata['upscaling']}]".strip()

    def number(value):
        return float(value) if isinstance(value, (int, float, np.number)) and np.isfinite(value) else None

    references = metadata.get("references")
    references = (references,) if isinstance(references, str) and references.strip() else tuple(references or ())
    doi = tuple(str(item) for item in (metadata.get("DOI") or ()) if str(item).strip())
    return encoding3d.Variable(unit=unit, standard_name=standard_name, kind=_KINDS.get(metadata.get("kind")),
                            role="derived" if derived else _ROLES.get(metadata.get("role"), "state"),
                            description=description, minimum=number(metadata.get("min_value")),
                            maximum=number(metadata.get("max_value")), references=tuple(str(r) for r in references),
                            doi=doi)


def _declare(table, metadata: dict, standard_names: dict, derived: set) -> None:
    for column in table.columns:
        name = column[len("mtg_"):] if column.startswith("mtg_") else column
        if column in table.reserved or column in LINK_COLUMNS or name not in metadata:
            continue
        table.variables[column] = _variable(name, metadata[name], standard_names, derived=name in derived)


def _json_meta(meta: dict) -> dict:
    out = {}
    for key, value in meta.items():
        if key == "dtype":
            value = {float: "float", int: "int", object: "object"}.get(value, str(value))
        elif key == "shape":
            value = list(value or ())
        elif isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float) and not math.isfinite(value):
            value = str(value)
        out[key] = value
    return out


def _restored_meta(meta: dict) -> dict:
    out = dict(meta)
    for key in ("location", "column", "encoding"):
        out.pop(key, None)
    if "dtype" in out:
        out["dtype"] = {"float": float, "int": int, "object": object}[out["dtype"]]
    if "shape" in out:
        out["shape"] = tuple(out["shape"])
    if isinstance(out.get("default"), str):
        out["default"] = float(out["default"])
    return out


# ── Columns ────────────────────────────────────────────────────────────────

def _empty_column(dtype, count: int, shape: tuple = ()):
    shape = (count,) + tuple(shape)
    if dtype == object:
        return np.full(shape, "", dtype=object)
    if np.issubdtype(np.dtype(dtype), np.floating):
        return np.full(shape, np.nan, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def _place(columns: dict, column: str, rows: np.ndarray, values: np.ndarray, count: int) -> None:
    """Write *values* at *rows* of the column (created with NaN / zeros elsewhere when new)."""
    values = np.asarray(values)
    if values.dtype == object:
        if not all(isinstance(value, str) for value in values.reshape(-1)):
            raise TypeError("only text object variables are exchanged")
    if column not in columns or columns[column].dtype != values.dtype or columns[column].shape[1:] != values.shape[1:]:
        columns[column] = _empty_column(values.dtype, count, values.shape[1:])
    columns[column][rows] = values


def _rows_of(ids: np.ndarray, wanted) -> np.ndarray:
    order = np.argsort(ids, kind="stable")
    position = np.searchsorted(ids[order], np.asarray(wanted, dtype=np.int64))
    rows = order[np.clip(position, 0, ids.size - 1)]
    if wanted is not None and len(wanted) and not np.array_equal(ids[rows], np.asarray(wanted, dtype=np.int64)):
        raise KeyError("ids not found in the table")
    return rows


def _round_trip(value):
    """*value* as JSON values, tuples, arrays, tables (pandas DataFrames), non-text keys and non-finite floats marked,
    so that _decoded gives it back; TypeError for what JSON cannot hold."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__float__": repr(value)}
    if isinstance(value, tuple):
        return {"__tuple__": [_round_trip(item) for item in value]}
    if isinstance(value, list):
        return [_round_trip(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"__array__": [_round_trip(item) for item in value.tolist()], "dtype": str(value.dtype)}
    if isinstance(value, dict):
        if all(isinstance(key, str) and not key.startswith("__") for key in value):
            return {key: _round_trip(item) for key, item in value.items()}
        return {"__dict__": [[_round_trip(key), _round_trip(item)] for key, item in value.items()]}
    if type(value).__name__ == "DataFrame" and hasattr(value, "to_dict"):      # a table (Adel's plants' tables)
        return {"__table__": {column: _round_trip(list(value[column])) for column in value.columns},
                "index": _round_trip(list(value.index))}
    raise TypeError(f"a {type(value).__name__} cannot be written as JSON")


def _encoded(value) -> str:
    return json.dumps(_round_trip(value), separators=(",", ":"), allow_nan=False)


def _decoded(text: str):
    def hook(item: dict):
        if "__tuple__" in item:
            return tuple(item["__tuple__"])
        if "__float__" in item:
            return float(item["__float__"])
        if "__array__" in item:
            return np.array(item["__array__"], dtype=item["dtype"])
        if "__dict__" in item:
            return {_hashable(key): value for key, value in item["__dict__"]}
        if "__table__" in item:
            import pandas as pd
            return pd.DataFrame(item["__table__"], index=item["index"])
        return item
    return json.loads(text, object_hook=hook)


def _hashable(key):
    return tuple(_hashable(item) for item in key) if isinstance(key, list) else key


_MISSING = object()


def _row_ranges(mask: np.ndarray) -> list:
    """The rows where *mask* holds, as the flat run-length list [start, stop, start, stop ...] of half-open ranges
    (the extras' absent_rows and none_rows: a few runs where vid lists ran to millions of values)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])))
    return edges.tolist()


def _ranges_mask(ranges, count: int) -> np.ndarray:
    """The mask of *count* rows holding the half-open ranges *ranges* (_row_ranges')."""
    bounds = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
    steps = np.zeros(count + 1, dtype=np.int64)
    np.add.at(steps, bounds[:, 0], 1)
    np.add.at(steps, bounds[:, 1], -1)
    return np.cumsum(steps[:count]) > 0


def _listed(present: np.ndarray, items: list) -> list:
    """A list over *present*'s rows: *items* (one per present row, in order) where present, None elsewhere."""
    out = np.empty(present.size, dtype=object)
    out[present] = np.fromiter(items, dtype=object, count=len(items))
    return out.tolist()


def _lookup(prop, vids: np.ndarray, plan: tuple = None) -> tuple:
    """(present, values) of the MPG property *prop* at *vids*: a mask, and the values (a float array, NaN where
    absent, for a float column; a list, None where absent, for another), read in bulk from its sorted keys. *plan*:
    the instances' plan of *vids* (Instances.plan), located once for every property read at them."""
    if isinstance(prop, _InstancedProperty):
        return prop.lookup(np.asarray(vids, dtype=np.int64), plan)
    if isinstance(prop, ArrayDict):
        present, values = prop.at(np.asarray(vids, dtype=np.int64))
        if values.dtype.kind == "f":
            out = np.full(present.size, np.nan)
            out[present] = values
            return present, out
        return present, _listed(present, values.tolist())
    get = prop.get
    values = [get(v, _MISSING) for v in vids.tolist()]
    present = np.array([value is not _MISSING for value in values], dtype=bool)
    return present, [None if value is _MISSING else value for value in values]


class _InstancedProperty:
    """An MPG property read with its instanced vertices' template values (MPG._property_at): a stand-in for the
    property in _lookup, ArrayDict-like when its values are numbers."""

    def __init__(self, g, name: str):
        self.g, self.name = g, name
        prop = g.properties().get(name)
        if prop is not None and len(prop):
            self.numeric = isinstance(prop, ArrayDict) and prop.dtype.kind == "f"
        else:
            self.numeric = all(template.properties[name][1].dtype != object
                               for template in g.instances.templates.values() if name in template.properties)

    def lookup(self, vids: np.ndarray, plan: tuple = None) -> tuple:
        """(present, values) at sorted unique *vids*, as _lookup gives them. *plan*: g.instances.plan(vids), the
        instanced vertices located once for every property (else located here): the same present and values as
        MPG._property_at's, merged in the vids' order without sorting them again."""
        if plan is None:
            plan = self.g.instances.plan(vids)
        explicit, values = self.g._explicit_property_mask(self.name, vids)
        has, template_values = self.g.instances.values(self.name, vids, plan)
        has &= ~explicit                                 # an entry of its own first (MPG._property_at_instanced)
        if not has.any():
            present, merged = explicit, values
        else:
            present = explicit | has
            rank = np.cumsum(present) - 1                 # each present vid's place among them (ids ascending)
            if values.dtype == object or template_values.dtype == object:
                merged = np.empty(int(rank[-1]) + 1, dtype=object)
                merged[rank[explicit]] = np.fromiter(values.tolist(), dtype=object, count=values.size)
                merged[rank[has]] = template_values[has]
            else:
                merged = np.empty(int(rank[-1]) + 1, dtype=np.float64)
                merged[rank[explicit]] = np.asarray(values, dtype=np.float64)
                merged[rank[has]] = template_values[has]
        if self.numeric:
            out = np.full(vids.size, np.nan)
            out[present] = np.asarray(merged, dtype=np.float64)
            return present, out
        return present, _listed(present, merged.tolist())


def _by_vid(mapping: dict, size: int) -> np.ndarray:
    """The integer values of *mapping* ({vid: value}: an MTG's dict) by vid, -1 for a missing or None value."""
    out = np.full(size, -1, dtype=np.int64)
    if mapping:
        keys = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
        values = np.array(list(mapping.values()), dtype=object)
        values[np.equal(values, None)] = -1
        out[keys] = values.astype(np.int64)
    return out


def _mpg_properties(g) -> dict:
    """{name: property} of *g*, with the template-only properties of its instances, each read with the instanced
    vertices' values (_InstancedProperty) when *g* has instances."""
    if not g._instanced():
        return dict(g.properties())
    names = list(g.properties()) + sorted({name for template in g.instances.templates.values()
                                           for name in template.properties} - set(g.properties()))
    return {name: _InstancedProperty(g, name) for name in names}


def _arraydict(keys: np.ndarray, values: np.ndarray) -> ArrayDict:
    """An ArrayDict of unique *keys* and their *values*, built from the arrays at once."""
    keys, values = np.asarray(keys, dtype=np.int64), np.asarray(values, dtype=np.float64)
    order = np.argsort(keys, kind="stable")
    out = ArrayDict(init_capacity=keys.size)
    out.order[:keys.size], out.arr[:keys.size] = keys[order], values[order]
    out.size = int(keys.size)
    out.vid2idx = None                                   # rebuilt at its first use
    return out


# ── MPGDataStructure → data structure ───────────────────────────────────────

def _scale_names(g) -> dict:
    from openalea.metafspm.coupling.declaration import scale_names
    return scale_names(g)


def _location_scales(ds) -> dict:
    """{DataStructure location: MPG scale} of a graph DataStructure's locations other than scalar."""
    g = ds.structure
    scales = {"node": g.scales.Compartment if ds._anatomy else ds._from_scale, "edge": g.scales.Connection}
    for name in ds._coarse_scale_names():
        scales[name] = getattr(g.scales, name)
    return scales


def _mpg_data_structure(identifier: str, ds, metadata: dict, standard_names: dict, geometry: bool, components=(),
                        display_meshes: bool = True, display_plants=None, display: str = "instances",
                        variables=None):
    encoding3d = _encoding3d()
    ds.flush_mtg()                     # the MPG holds the variables' current values, as ds.mtg gives it
    g = ds.structure
    names = _scale_names(g)
    arrays = g.topology_arrays()
    anchors = set(np.flatnonzero(arrays["is_anchor"]).tolist()) | {g.root}
    # the MTG's own scale, complex and parent of each vertex, by vid (-1: none), read from its dicts in bulk
    size = max(arrays["scale"].size, *(max(mapping, default=-1) + 1 for mapping in (g._scale, g._complex, g._parent)))
    own_scale, own_complex, own_parent = (_by_vid(mapping, size) for mapping in (g._scale, g._complex, g._parent))
    explicit = np.flatnonzero(own_scale >= 0)
    explicit = explicit[~np.isin(explicit, list(anchors))]
    by_scale = {scale: explicit[own_scale[explicit] == scale] for scale in np.unique(own_scale[explicit]).tolist()}
    instanced = g._instanced()
    if instanced:                                  # the instanced vertices, as their copies would be listed
        blocks = [np.arange(f, f + n, dtype=np.int64)
                  for f, n in zip(g.instances.first.tolist(), g.instances.size.tolist())]
        virtual = np.concatenate(blocks)
        for scale in np.unique(arrays["scale"][virtual]).tolist():
            here = virtual[arrays["scale"][virtual] == scale]
            by_scale[scale] = np.union1d(by_scale.get(scale, np.zeros(0, dtype=np.int64)), here)
        _, virtual_complex, virtual_parent = g.instances.structure(virtual)
        anchored = virtual_complex <= -2
        virtual_complex[anchored] = [g.scales.anchors[int(-2 - c)] for c in virtual_complex[anchored].tolist()]
        own_complex[virtual], own_parent[virtual] = virtual_complex, virtual_parent     # an instance's links first
    connection = g.scales.Connection
    translator = dict(g.labels.translator)
    properties = _mpg_properties(g)                # the columns (g.property() builds a mapping)
    label_property, edge_property = properties.get("label", {}), properties.get("edge_type", {})
    extras = {"class": "MPGDataStructure", "construction": ds._construction(), "last_id": int(g._id),
              "anchors": sorted(int(v) for v in anchors), "scales": {}, "label_codes": {}, "properties": {},
              "absent_rows": {}, "none_rows": {}, "variables": {},
              "tracked": sorted(ds.__dict__.get("_mtg_tracked", {})),
              "aliases": ds.aliases()}
    tables = {}

    for scale, vids in sorted(by_scale.items()):
        vids = np.asarray(vids, dtype=np.int64)
        # read-only for the properties' reads: the DataStructure's columns share their positions in their location's
        # ids (columns._positions_shared), and the instanced vertices are located once for every property (plan)
        read = vids.view()
        read.flags.writeable = False
        plan = g.instances.plan(read) if instanced else None
        name = names[scale]
        relation = scale == connection
        reserved = _RELATION_RESERVED if relation else _GRAPH_RESERVED
        raw_complex, raw_parent = own_complex[vids], own_parent[vids]
        columns = {"id": vids}
        present, codes = _lookup(label_property, read, plan)
        if not isinstance(codes, np.ndarray) or codes.dtype != np.float64:
            codes = np.array([np.nan if code is None else code for code in codes], dtype=float)
        if present.any():
            label_column = np.full(vids.size, "_none", dtype=object)
            for code in np.unique(codes[present]):
                label = translator.get(int(code), f"label_{int(code)}")
                known = extras["label_codes"].setdefault(name, {})
                if label in known and known[label] != code:     # the same name for two codes at this scale
                    label = f"{label}_{int(code)}"
                known[label] = int(code) if float(code).is_integer() else float(code)
                label_column[present & (codes == code)] = label
            columns["relation" if relation else "label"] = label_column
        if relation:
            columns["source"] = _lookup(properties.get("n_id_a", {}), read, plan)[1].astype(np.int64)
            columns["target"] = _lookup(properties.get("n_id_b", {}), read, plan)[1].astype(np.int64)
        else:
            same_scale = (raw_parent >= 0) & (arrays["scale"][np.maximum(raw_parent, 0)] == scale)
            resolved = arrays["complex"][vids]
            owned = (resolved >= 0) & ~np.isin(resolved, list(anchors))
            # the containing entity: the complex, or for a vertex anchored to its scale's anchor, its parent at a
            # coarser scale (a wall's segment)
            complex_ = np.where(owned, resolved, np.where((raw_parent >= 0) & ~same_scale, raw_parent, NO_ID))
            if scale != min(by_scale):
                columns["complex"] = complex_.astype(np.int64)
            columns["parent"] = np.where(same_scale, raw_parent, NO_ID).astype(np.int64)
            edge_codes = arrays["edge_type"][vids]
            edge_types = np.full(vids.size, "none", dtype=object)
            edge_types[same_scale] = np.where(edge_codes[same_scale] == 3, "branch", "successor")
            columns["edge_type"] = edge_types
        columns["metafspm_complex"], columns["metafspm_parent"] = raw_complex, raw_parent
        raw_edges = np.array(_lookup(edge_property, read, plan)[1] if isinstance(edge_property, _InstancedProperty)
                             else _lookup(edge_property, read)[1] if isinstance(edge_property, ArrayDict)
                             else [edge_property.get(v) for v in vids.tolist()], dtype=object)
        if not set(raw_edges.tolist()) <= {None, "/", "<", "+"}:
            raise ValueError(f"{name}: edge types other than '/', '<' and '+' cannot be exchanged")
        edge_type_codes = np.zeros(vids.size, dtype=np.uint8)
        for edge, code in (("/", 1), ("<", 2), ("+", 3)):
            edge_type_codes[raw_edges == edge] = code
        columns["metafspm_edge_type"] = edge_type_codes

        for property_name, prop in properties.items():
            if property_name in ("label", "edge_type") or (relation and property_name in ("n_id_a", "n_id_b")):
                continue
            present, values = _lookup(prop, read, plan)
            if not present.any():
                continue
            column = f"mtg_{property_name}" if property_name in reserved or property_name in LINK_COLUMNS \
                else property_name
            if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", column):
                warnings.warn(f"MPG property '{property_name}' is not exchanged: its name is not an identifier")
                continue
            numeric = (isinstance(prop, ArrayDict) and prop.dtype.kind == "f") or \
                (isinstance(prop, _InstancedProperty) and prop.numeric)          # a float column; else its values
            if numeric:
                kind, data = "float64", values
            else:
                assigned = [value for value in values if value is not None]
                if not assigned:
                    kind, data = None, None
                elif all(isinstance(value, (bool, np.bool_)) for value in assigned):
                    kind, data = "bool", np.array([bool(value) for value in values])
                elif all(isinstance(value, (int, np.integer)) and not isinstance(value, bool) for value in assigned):
                    kind, data = "int64", np.array([0 if value is None else value for value in values], dtype=np.int64)
                elif all(isinstance(value, (int, float, np.number)) for value in assigned):
                    kind = "float64"
                    data = np.array([np.nan if value is None else value for value in values], dtype=np.float64)
                elif all(isinstance(value, str) for value in assigned):
                    kind, data = "str", np.array(["" if value is None else value for value in values], dtype=object)
                else:
                    try:                                    # lists, tuples, records: as JSON text
                        kind = "json"
                        data = np.array(["" if value is None else _encoded(value) for value in values], dtype=object)
                    except TypeError as error:
                        warnings.warn(f"MPG property '{property_name}' is not exchanged at {name}: {error}")
                        continue
            entry = extras["properties"].setdefault(property_name, {
                "container": "ArrayDict" if numeric else "dict", "column": column})
            entry.setdefault("dtypes", {})[name] = kind
            if not present.all():                       # the rows (of the scale's vids) without the property
                extras["absent_rows"][f"{name}.{property_name}"] = _row_ranges(~present)
            if not numeric:                             # and those holding it as None
                none = np.array([here and value is None for here, value in zip(present.tolist(), values)], dtype=bool)
                if none.any():
                    extras["none_rows"][f"{name}.{property_name}"] = _row_ranges(none)
            if data is not None:
                columns[column] = data
        tables[scale] = columns
        extras["scales"][name] = scale

    # The DataStructure's variables, at the scale of their location
    scale_of = _location_scales(ds)
    for location, store in ds._var_stores().items():
        if location == "scalar" or not store:
            continue
        scale = scale_of[location]
        if scale not in tables:
            raise ValueError(f"{identifier}: variables at {location} but no MPG vertex at its scale")
        columns = tables[scale]
        ids = np.asarray(ds.entity_ids(location), dtype=np.int64)
        rows = _edge_rows(columns, ids) if location == "edge" else _rows_of(columns["id"], ids)
        reserved = _RELATION_RESERVED if scale == connection else _GRAPH_RESERVED
        # the instanced vertices (a reduced or instanced anatomy's): their values are the DataStructure's only, the
        # MPG giving their template's
        explicit = (~np.asarray(ds._instanced_at(location), dtype=bool)
                    if instanced and location in ("node", "edge") and hasattr(ds, "_instanced_at") else None)
        for name, values in store.items():
            column = f"metafspm_{name}" if name in reserved or name in LINK_COLUMNS else name
            try:
                values, encoding = _object_values(values)
            except TypeError as error:
                warnings.warn(f"variable '{name}' is not exchanged: {error}")
                continue
            if column in columns and not _holds(columns[column], rows, values) and not (
                    explicit is not None and _holds(columns[column], rows[explicit], values[explicit])):
                column = f"metafspm_{name}"         # an MPG property of that name holds other values or text
            _place(columns, column, rows, values, columns["id"].size)
            extras["variables"][name] = dict(_json_meta(ds._variable_meta().get(name, {})), location=location,
                                             **({"column": column} if column != name else {}),
                                             **({"encoding": encoding} if encoding else {}))
    drawing = geometry and display_meshes
    instanced_by = _instancing_components(ds, components) if drawing and display == "instances" else []
    if drawing and (display == "full" or (display == "instances" and not instanced_by)) and \
            any(callable(getattr(c, "display_triangles", None)) for c in components):
        _display_interiors(ds, tables, extras, names, translator, display_plants)   # the reduced sections' interiors
    scalars = _scalar_table(ds, extras)
    extras["tracked"] = [name for name in extras["tracked"] if name in extras["variables"]]
    extras.update(_anchor_and_empty_properties(g, anchors))

    entity_sets, relations = [], []
    for scale, columns in sorted(tables.items()):
        if scale == connection:
            ends = np.concatenate([columns["source"], columns["target"]])
            endpoints = set(arrays["scale"][ends[(ends >= 0) & (ends < arrays["scale"].size)]].tolist())
            endpoint = names[endpoints.pop()] if len(endpoints) == 1 else names[g.scales.Compartment]
            relations.append(encoding3d.Relation(names[scale], columns=columns, source_set=endpoint,
                                              target_set=endpoint))
        else:
            base_scales = encoding3d.model.BASE_SCALES             # a base scale at its position: declared as such
            declared = names[scale] if names[scale] in base_scales and base_scales.index(names[scale]) + 1 == int(scale) \
                else None
            entity_sets.append(encoding3d.EntitySet(names[scale], columns=columns, rank=int(scale), scale=declared))
    derived = set(ds.derived())
    for table in entity_sets + relations + ([scalars] if scalars is not None else []):
        _declare(table, metadata, standard_names, derived)
        for column in ("isanchor", "scale"):
            if column in table.columns and column not in table.variables:
                table.variables[column] = encoding3d.Variable(unit="1", kind="descriptor", role="setting",
                                                           description=f"metafspm: the MPG property {column}")
        if DISPLAY_ONLY in table.columns:
            table.variables[DISPLAY_ONLY] = encoding3d.Variable(
                unit="1", kind="descriptor", role="setting",
                description="metafspm: a row for display only, a reduced section's interior vertex (its values: the "
                            "reduced system's unknown recovered, its full template's properties, NaN otherwise); "
                            "not a vertex of the MPG")
        for column in LINK_COLUMNS:
            if column in table.columns:
                table.variables[column] = encoding3d.Variable(
                    unit="1", kind="descriptor", role="setting",
                    description={"metafspm_complex": "metafspm: the MPG's own complex of the vertex (-1: none)",
                                 "metafspm_parent": "metafspm: the MPG's own parent of the vertex, at any scale "
                                                    "(-1: none)",
                                 "metafspm_edge_type": "metafspm: the MPG's edge type (0 none, 1 '/', 2 '<', "
                                                       "3 '+')"}[column])
    reduced_nodes = _reduced_nodes(ds) if instanced_by or (drawing and display is None) else None
    meshes = _meshes(ds, entity_sets, scale_of, names, components, display_meshes, display_plants,
                     reduced_nodes) if geometry else []
    instance_sets = _instance_sets(ds, instanced_by, entity_sets, scale_of, names, translator, extras, metadata,
                                   standard_names, display_plants, variables) if instanced_by else []
    if instance_sets:
        extras["instances"] = {instance_set.name: {"owners": instance_set.owners,
                                                   "derived": [field.name for field in instance_set.derived]}
                               for instance_set in instance_sets}
    return encoding3d.DataStructure(identifier, kind="graph", entity_sets=entity_sets, relations=relations,
                                 scalars=scalars, meshes=meshes, platform=PLATFORM, model="MPGDataStructure",
                                 extras={EXTRAS: extras}, instance_sets=instance_sets)


def _plants_mask(ds, plants, location: str = "node") -> np.ndarray:
    """Whether each entity at *location* of *ds* belongs to one of the Plant vids *plants*."""
    plant_ids = np.asarray(ds.entity_ids("Plant"), dtype=np.int64)
    owner = np.asarray(ds.owner("Plant") if location == "node" else ds.owner("Plant", of=location), dtype=np.int64)
    return (owner >= 0) & np.isin(plant_ids[np.maximum(owner, 0)], np.asarray(list(plants), dtype=np.int64))


def _display_interiors(ds, tables: dict, extras: dict, names: dict, translator: dict, plants=None) -> None:
    """
    The reduced sections' interiors as display rows (the frame's display meshes draw the sections whole): each reduced
    group's full template vertices its reduced template does not hold (its eliminated Compartments, and the Cells and
    Layers holding them; not its Connections), appended to the tables of their scales under their display ids
    (solve.reduction.display_ids), marked by the column metafspm_display_only, their complex and parent links resolved
    to the group's vertices (kept or displayed) and to its owner. Their values: the reduced system's unknown recovered
    from the kept nodes' (Reduction.recover_interiors: exact for the linear Kron reduction), their full template's
    properties, NaN (or the column's empty value) otherwise. from_frame leaves them out: the DataStructure stays
    reduced. *plants*: the Plant vids whose sections are drawn (None: all), the others' interiors not recovered.
    """
    from openalea.metafspm.data_structure.instances import ANCHOR, OWNER
    from openalea.metafspm.solve.reduction import display_ids
    g = ds.structure
    reduction = getattr(g, "reduction", None)
    if reduction is None or not g._instanced():
        return
    groups = reduction.reduced_groups(ds)
    if plants is not None and groups:
        drawn = _plants_mask(ds, plants, reduction.location)
        groups = {i: group for i, group in groups.items() if drawn[i]}
    if not groups:
        return
    if reduction.systems and reduction.instance is not None and not reduction.library:
        recovered, unknown = reduction.recover_interiors(ds, groups=list(groups)), reduction.unknown()
    else:                                           # no system bound: the interiors' structure, their values NaN
        recovered, unknown = {i: (owner, key, reduction.full_template(key), np.zeros(0, dtype=np.int64),
                                  np.zeros(0)) for i, (owner, key) in groups.items()}, None
    unknown_column = extras["variables"].get(unknown, {}).get("column", unknown) if unknown else None
    connection = g.scales.Connection
    parts = {}                                      # {scale: [(ids, scale, complex, parent, template, rows, values)]}
    for owner, key, full, interior_rows, values in recovered.values():
        first, _ = g.instances.block(owner)
        held = reduction.full_rows(key)
        index_in_reduced = np.full(full.size, -1, dtype=np.int64)
        index_in_reduced[held] = np.arange(held.size)
        ids_of_row = np.where(index_in_reduced >= 0, first + index_in_reduced,
                              display_ids(owner, np.arange(full.size)))
        added = np.flatnonzero((index_in_reduced < 0) & (full.scale != connection))
        unknown_at = np.full(full.size, np.nan)
        unknown_at[np.asarray(interior_rows, dtype=np.int64)] = values

        def resolved(codes, rows):
            codes = codes[rows]
            out = np.where(codes >= 0, ids_of_row[np.maximum(codes, 0)], NO_ID)
            out = np.where(codes == OWNER, owner, out)
            anchors = np.array([g.scales.anchors[int(scale)] for scale in full.scale[rows]], dtype=np.int64) \
                if rows.size else np.zeros(0, dtype=np.int64)
            return np.where(codes == ANCHOR, anchors, out)

        for scale in np.unique(full.scale[added]).tolist():
            rows = added[full.scale[added] == scale]
            parts.setdefault(scale, []).append((ids_of_row[rows], resolved(full.complex, rows),
                                                resolved(full.parent, rows), full, rows, unknown_at[rows]))
    arrays = g.topology_arrays()
    display_id_list = [item[0] for items in parts.values() for item in items]
    display_scale_list = [np.full(item[0].size, scale, dtype=np.int64) for scale, items in parts.items()
                          for item in items]
    display_order = np.concatenate(display_id_list) if display_id_list else np.zeros(0, dtype=np.int64)
    display_scales = np.concatenate(display_scale_list) if display_scale_list else np.zeros(0, dtype=np.int64)
    sort = np.argsort(display_order)
    display_order, display_scales = display_order[sort], display_scales[sort]

    def scale_at(vids):
        """The scale of each of *vids* (a display id's, an MPG vertex's; -1 for none)."""
        out = np.full(vids.size, -1, dtype=np.int64)
        real = (vids >= 0) & (vids < arrays["scale"].size)
        out[real] = arrays["scale"][vids[real]]
        if display_order.size:
            position = np.searchsorted(display_order, vids).clip(0, display_order.size - 1)
            shown = display_order[position] == vids
            out[shown] = display_scales[position[shown]]
        return out
    counts = {}
    for scale, items in sorted(parts.items()):
        if scale not in tables:
            continue                                # no vertex of the MPG at that scale: not drawn by scale
        set_name = names[scale]
        columns = tables[scale]
        ids = np.concatenate([item[0] for item in items])
        raw_complex = np.concatenate([item[1] for item in items])
        raw_parent = np.concatenate([item[2] for item in items])

        parent_scale = scale_at(raw_parent)
        same_scale = (raw_parent >= 0) & (parent_scale == scale)
        anchored = np.isin(raw_complex, list(g.scales.anchors.values()))
        complex_ = np.where(~anchored & (raw_complex >= 0), raw_complex,
                            np.where((raw_parent >= 0) & ~same_scale, raw_parent, NO_ID))
        n = ids.size
        new = {"id": ids, "complex": complex_.astype(np.int64),
               "parent": np.where(same_scale, raw_parent, NO_ID).astype(np.int64),
               "edge_type": np.array(["successor" if same else "none" for same in same_scale], dtype=object),
               "metafspm_complex": raw_complex, "metafspm_parent": raw_parent,
               "metafspm_edge_type": np.zeros(n, dtype=np.uint8)}

        def template_values(name):
            """(present, values) of the template property *name* at the display rows (object values)."""
            present, out = np.zeros(n, dtype=bool), np.empty(n, dtype=object)
            start = 0
            for _, _, _, full, rows, _ in items:
                stored = full.properties.get(name)
                if stored is not None:
                    where, values = stored
                    position = np.searchsorted(where, rows).clip(0, max(where.size - 1, 0))
                    here = (where[position] == rows) if where.size else np.zeros(rows.size, dtype=bool)
                    present[start:start + rows.size] = here
                    out[start:start + rows.size][here] = list(np.asarray(values)[position[here]])
                start += rows.size
            return present, out

        present, codes = template_values("label")
        if "label" in columns:
            known = extras["label_codes"].setdefault(set_name, {})
            name_of = {float(code): label for label, code in known.items()}
            numbers = np.array([float(code) if here else np.nan for here, code in zip(present, codes)])
            labels = np.full(n, "_none", dtype=object)
            for code in np.unique(numbers[present]).tolist():
                label = name_of.get(code)
                if label is None:                    # a label only the interiors have
                    label = translator.get(int(code), f"label_{int(code)}")
                    if label in known:
                        label = f"{label}_{int(code)}"
                    known[label] = int(code) if float(code).is_integer() else float(code)
                    name_of[code] = label
                labels[numbers == code] = label
            new["label"] = labels
        for property_name, entry in extras["properties"].items():
            column, kind = entry["column"], entry.get("dtypes", {}).get(set_name)
            if column not in columns or column in new or kind is None:
                continue
            here, values = template_values(property_name)
            if not here.any():
                continue
            filled = _empty_column(columns[column].dtype, n, columns[column].shape[1:])
            if kind == "float64" and filled.ndim == 1:
                filled[here] = np.asarray(values[here].tolist(), dtype=np.float64)
            else:
                convert = {"float64": float, "int64": int, "bool": bool, "str": str, "json": _encoded}[kind]
                for k in np.flatnonzero(here).tolist():
                    filled[k] = convert(values[k])
            new[column] = filled
        if unknown_column is not None and unknown_column in columns and \
                np.issubdtype(columns[unknown_column].dtype, np.floating):
            new[unknown_column] = np.concatenate([item[5] for item in items])
        for column, old in list(columns.items()):
            added = new[column] if column in new else _empty_column(old.dtype, n, old.shape[1:])
            columns[column] = np.concatenate([old, added])
        columns[DISPLAY_ONLY] = np.concatenate([np.zeros(columns["id"].size - n, dtype=bool), np.ones(n, dtype=bool)])
        counts[set_name] = int(n)
    if counts:
        extras["display_only"] = {"rows": counts, "column": DISPLAY_ONLY, "values": unknown}


def _reduction_of(ds):
    """The Reduction of *ds*'s MPG with reduced groups (None: none)."""
    g = ds.structure
    reduction = getattr(g, "reduction", None)
    if reduction is None or not g._instanced() or reduction.group_table(ds) is None:
        return None
    return reduction


def _instancing_components(ds, components) -> list:
    """The *components* drawing *ds*'s reduced groups as instances (display_instances), [] without a reduced group."""
    if _reduction_of(ds) is None:
        return []
    return [component for component in components if callable(getattr(component, "display_instances", None))]


def _reduced_nodes(ds):
    """The node mask of *ds*'s reduced groups (None without a reduction): their kept nodes."""
    reduction = _reduction_of(ds)
    if reduction is None:
        return None
    table = reduction.group_table(ds)
    owner = np.asarray(ds.owner(reduction.location), dtype=np.int64)
    reduced = np.zeros(len(ds.entity_ids(reduction.location)), dtype=bool)
    reduced[table.index] = True
    return (owner >= 0) & reduced[np.maximum(owner, 0)]


def _instance_sets(ds, components, entity_sets: list, scale_of: dict, names: dict, translator: dict, extras: dict,
                   metadata: dict, standard_names: dict, plants=None, variables=None) -> list:
    """
    The reduced groups of *ds* as encoding3d instance sets (specification §12), one per component drawing them
    (display_instances: a TemplateShapes), named after it: each group an instance of its template (a shared part:
    the component's shape in template coordinates, on the full template's nodes, with their constant properties),
    placed by its transform, owned by its group's vertex (an entity of the reduction's location); its vector the kept
    nodes' values of each reduced system's unknown, then 1; a derived field per system, the unknown on the template's
    nodes, the weighted sum of its archetypes' recovery operators (shared parts: Archetype.recovery_operator, a class blend's two
    archetypes weighted) applied to it, as Reduction.recover_interiors recovers them (the kept nodes their own values).
    The parts are kept on the Reduction (built once per template and archetype, as long as the archetype is held). *plants*:
    the Plant vids whose groups are drawn (None: all). *variables*: a variables filter (filter_frame's), the
    templates' node columns kept by it as the node set's (their id and label always).
    """
    encoding3d = _encoding3d()
    reduction = _reduction_of(ds)
    g = ds.structure
    groups = reduction.reduced_groups(ds)
    if plants is not None and groups:
        drawn = _plants_mask(ds, plants, reduction.location)
        groups = {i: group for i, group in groups.items() if drawn[i]}
    if not groups:
        return []
    owner_scale = names[int(g.topology_arrays()["scale"][next(iter(groups.values()))[0]])]   # the location's set
    node_set = names[scale_of["node"]]                     # the reduced system's nodes: the templates' rows
    kept = _table_variables(variables, node_set) if variables is not None else None
    kept = None if kept is None or isinstance(kept, Excluding) else frozenset(kept)
    bound = bool(reduction.systems) and reduction.instance is not None and not reduction.library
    terms = [reduction.recovery_terms(ds, groups=list(groups), system=system) for system in reduction.systems] \
        if bound else []
    terms = [item for item in terms if item is not None]
    full_of = {int(owner): reduction.full_template(key) for owner, key in groups.values()}
    parts_cache = reduction.__dict__.setdefault("_display_parts", {})
    held = set(reduction._archetypes)
    for cached in [key for key in parts_cache if key[0] == "operator" and key[1] not in held]:
        del parts_cache[cached]                    # an archetype released: its operator too
    out = []
    for component in components:
        component_name = type(component).__name__
        shapes = component.display_instances(np.asarray(sorted(full_of), dtype=np.int64))
        if not len(shapes):
            continue
        unknown_owner = sorted(set(shapes.owners.tolist()) - set(full_of))
        if unknown_owner:
            raise ValueError(f"{component_name}.display_instances drew groups that are not reduced ones: "
                             f"{unknown_owner[:5]}")
        parts, part_index = [], {}

        def part_of(key, build):
            if key not in part_index:
                if key not in parts_cache:
                    parts_cache[key] = build()
                part_index[key] = len(parts)
                parts.append(parts_cache[key])
            return part_index[key]
        template_index = np.empty(len(shapes), dtype=np.int64)
        for position, (owner, shape_key) in enumerate(zip(shapes.owners.tolist(), shapes.shape_of)):
            full = full_of[owner]
            template_index[position] = part_of(
                ("template", component_name, shape_key, full.key, kept),
                lambda: _template_part(full, shapes.shapes[shape_key], node_set, component_name, translator, g,
                                       metadata, standard_names, kept))
        columns = {"id": shapes.owners.copy(), "owner": shapes.owners.copy(), "template": template_index,
                   "transform": encoding3d.column_major(shapes.transforms)}
        vectors, derived = {}, []
        for item in terms:
            row_of = dict(zip(item.owner.tolist(), range(item.owner.size)))
            rows = np.array([row_of.get(owner, -1) for owner in shapes.owners.tolist()], dtype=np.int64)
            operator_of = np.array([part_of(("operator", key, item.system), lambda archetype=archetype: _operator_part(archetype))
                                    for key, archetype in zip(item.archetype_keys, item.archetypes)], dtype=np.int64)
            corners = item.archetype.shape[1]
            operators = np.full((rows.size, corners), NO_ID, dtype=np.int64)
            weights = np.zeros((rows.size, corners))
            found = rows >= 0
            chosen = item.archetype[rows[found]]
            operators[found] = np.where(chosen >= 0, operator_of[np.maximum(chosen, 0)], NO_ID)
            weights[found] = item.weights[rows[found]]
            unknown_column = extras["variables"].get(item.unknown, {}).get("column", item.unknown)
            vector = f"{unknown_column}_kept"
            vectors[vector] = encoding3d.Ragged.of([item.inputs[row] if row >= 0 else np.zeros(0)
                                                    for row in rows.tolist()])
            derived.append(encoding3d.DerivedField(
                unknown_column, entity_set=node_set, input=vector, operators=operators, weights=weights,
                variable=_variable(item.unknown, metadata.get(item.unknown, {}), standard_names, derived=True)))
        out.append(encoding3d.InstanceSet(component_name, owners=owner_scale, parts=parts, columns=columns,
                                          vectors=vectors, derived=derived))
    return out


def _template_part(full, shape, node_set: str, mesh_name: str, translator: dict, g, metadata: dict = None,
                   standard_names: dict = None, kept=None):
    """The encoding3d Template of a reduced group's full template *full* drawn by *shape* (Triangles in template
    coordinates, each triangle's owner a full template row): its entity set *node_set*, the template's nodes (its
    Compartments, ascending: the derived fields' rows), ids their template rows, with their constant properties
    (label names, numbers, text; *kept*: those named only, None: all); its mesh *mesh_name*, the vertices shared by
    its triangles."""
    encoding3d = _encoding3d()
    rows = np.flatnonzero(full.scale == g.scales.Compartment)
    position = np.searchsorted(rows, shape.owner).clip(0, max(rows.size - 1, 0))
    if not rows.size or (rows[position] != shape.owner).any():
        raise ValueError(f"mesh '{mesh_name}': a template shape's triangle on a template row that is not one of its "
                         f"nodes (Compartments): {sorted(set(shape.owner[rows[position] != shape.owner].tolist()))[:5]}")
    columns = {"id": rows.astype(np.int64)}
    from openalea.metafspm.data_structure.instances import LINKS
    for name, (where, values) in sorted(full.properties.items()):
        if name in LINKS or name in ("scale", "isanchor") or (name in _GRAPH_RESERVED and name != "label") or \
                (kept is not None and name != "label" and name not in kept):
            continue
        at = np.searchsorted(rows, where).clip(0, max(rows.size - 1, 0))
        here = rows[at] == where if np.size(where) else np.zeros(0, dtype=bool)
        if not here.any():
            continue
        values = np.asarray(values)[here]
        if name == "label":
            labels = np.full(rows.size, "_none", dtype=object)
            labels[at[here]] = [translator.get(int(code), f"label_{int(code)}") for code in values.tolist()]
            columns["label"] = labels
        elif values.dtype != object and values.dtype.kind in "fiub":
            column = np.full(rows.size, np.nan)
            column[at[here]] = values.astype(float)
            columns[name] = column
        elif all(isinstance(value, str) for value in values.tolist()):
            column = np.full(rows.size, "", dtype=object)
            column[at[here]] = values
            columns[name] = column
    positions = shape.vertices.reshape(-1, 3)
    feature_rows = {node_set: np.repeat(position, 3).astype(np.int64)}
    positions, feature_rows, indices = _shared_vertices(positions, feature_rows)
    nodes = encoding3d.EntitySet(node_set, columns=columns)
    _declare(nodes, metadata or {}, standard_names or {}, set())
    return encoding3d.Template(entity_sets=[nodes],
                               meshes=[encoding3d.Mesh(mesh_name, positions=positions, feature_rows=feature_rows,
                                                       indices=indices)])


def _operator_part(archetype):
    """The encoding3d Operator of *archetype*'s recovery (Archetype.recovery_operator): dense, or its sparse system solved by
    the reader."""
    encoding3d = _encoding3d()
    found = archetype.recovery_operator()
    return encoding3d.Operator.dense(found[1]) if found[0] == "dense" else encoding3d.Operator.solve(found[1],
                                                                                                    found[2])


def _object_values(values) -> tuple:
    """(values, encoding) of a variable: object values other than text become JSON text (encoding "json")."""
    values = np.asarray(values)
    if values.dtype != object:
        return values, None
    flat = values.reshape(-1)
    if all(isinstance(value, str) for value in flat):
        return values, None
    return np.array([_encoded(value) for value in flat], dtype=object).reshape(values.shape), "json"


def _holds(column: np.ndarray, rows: np.ndarray, values: np.ndarray) -> bool:
    """Whether *column* already holds *values* at *rows*, with the same dtype (a variable and the MPG property it is
    written back to share their column then)."""
    if column.dtype != values.dtype or column.shape[1:] != values.shape[1:]:
        return False
    if values.dtype == object:
        return column[rows].tolist() == values.tolist()
    return np.ascontiguousarray(column[rows]).tobytes() == np.ascontiguousarray(values).tobytes()


def _scalar_table(ds, extras: dict):
    columns = {}
    for name, values in ds._var_stores().get("scalar", {}).items():
        try:
            values, encoding = _object_values(values)
        except TypeError as error:
            warnings.warn(f"variable '{name}' is not exchanged: {error}")
            continue
        columns[name] = np.asarray(values).reshape((1,) + np.shape(values))
        extras["variables"][name] = dict(_json_meta(ds._variable_meta().get(name, {})), location="scalar",
                                         **({"encoding": encoding} if encoding else {}))
    return _encoding3d().Table("scalars", columns=columns) if columns else None


def _anchor_and_empty_properties(g, anchors: set) -> dict:
    """What the tables cannot hold: the property values of the scale anchors that differ from a new MPG's, the
    properties without any value, and those of a new MPG that this one lacks."""
    fresh = MPG()
    anchor_values, empty = {}, {}
    for name, prop in g.properties().items():
        container = "ArrayDict" if isinstance(prop, ArrayDict) and prop.dtype.kind == "f" else "dict"   # an object column
        if len(prop) == 0:
            empty[name] = container
            continue
        default = fresh.properties().get(name, {})
        changed = {}
        for vid in anchors:
            here, there = vid in prop, vid in default
            if here and (not there or not _same_value(prop[vid], default[vid])):
                changed[str(vid)] = _encoded(prop[vid])
            elif there and not here:
                changed[str(vid)] = None
        if changed:
            anchor_values[name] = {"container": container, "values": changed}
    missing = sorted(set(fresh.properties()) - set(g.properties()))
    return {"anchor_properties": anchor_values, "empty_properties": empty, "missing_properties": missing}


def _same_value(a, b) -> bool:
    try:
        return _encoded(a) == _encoded(b)
    except TypeError:
        return a is b


def _edge_rows(columns: dict, ids: np.ndarray) -> np.ndarray:
    """Rows of the Connection relation holding the DataStructure's edges *ids*: their own vids (anatomy mode), or
    their child vids (segment mode: the Connection ending at that vertex)."""
    try:
        return _rows_of(columns["id"], ids)
    except KeyError:
        return _rows_of(columns["target"], ids)


def _meshes(ds, entity_sets: list, scale_of: dict, names: dict, components=(), display_meshes: bool = True,
            display_plants=None, reduced_nodes=None) -> list:
    """The DataStructure's triangles (its geometry provider's, those light models get), and with *display_meshes*
    each component's display_triangles() (shapes drawn for display only, e.g. GRANAP's sections; those of the Plant
    vids *display_plants* only when given), one mesh each, its vertices shared by its triangles. *reduced_nodes*: a
    node mask, the nodes of the reduced groups not drawn here (drawn as instances, or not at all: display None), the
    triangles of no node (a reduced group's interior) left out with them."""
    drawn = []
    # the plants' triangles (those light models get): on an anatomy-mode DataStructure its SubOrgans' (the view at
    # its from_scale, where the architectures draw them), their owners SubOrgans; the Compartments are not drawn
    drawn_on = ds.at(ds._scale_name(ds.from_scale)) if getattr(ds, "_anatomy", False) and hasattr(ds, "at") else ds
    try:
        drawn.append(("triangles", drawn_on, drawn_on.triangles(), False))
    except (ValueError, NotImplementedError):
        pass
    where = None if display_plants is None else _plants_mask(ds, display_plants)
    if reduced_nodes is not None:
        where = ~reduced_nodes if where is None else where & ~reduced_nodes
    for component in components if display_meshes else ():
        display = getattr(component, "display_triangles", None)
        if callable(display):
            triangles = display() if where is None else display(where=where)
            if reduced_nodes is not None and len(triangles):           # whatever the component drew of them
                owner = triangles.owner
                triangles = triangles.select(np.flatnonzero((owner >= 0) & ~reduced_nodes[np.maximum(owner, 0)]))
            drawn.append((type(component).__name__, ds, triangles, True))
    return [_mesh_of(on, entity_sets, scale_of, names, name, triangles, shared=shared)
            for name, on, triangles, shared in drawn if len(triangles)]


def _mesh_of(ds, entity_sets: list, scale_of: dict, names: dict, name: str, triangles, shared: bool = False):
    """The mesh of *triangles*, owned by the nodes of *ds* (a DataStructure, or a view of it at a coarse location:
    an anatomy-mode plant's SubOrgans), each vertex with the rows of its owner and of its owner's containing
    entities; *shared*: the vertices of equal position and rows written once, the triangles by their indices (a
    display shape's polygons: about a third of the vertices)."""
    encoding3d = _encoding3d()
    sets = {entity_set.name: entity_set for entity_set in entity_sets}
    node_ids = np.asarray(ds.entity_ids("node"), dtype=np.int64)
    owners = np.repeat(triangles.owner, 3)
    shown = owners < 0                                      # display rows' triangles (a reduced section's interior)
    if shown.any() and triangles.owner_ids is None:
        raise ValueError(f"mesh '{name}': triangles without owner (-1) and no owner_ids")
    feature_rows = {}
    at = ds.__dict__.get("_at")                             # a view: its nodes the entities of that location
    node_set = sets[names[scale_of[at if at is not None else "node"]]]
    rows = np.full(owners.size, NO_ID, dtype=np.int64)
    rows[~shown] = _rows_of(node_set.columns["id"], node_ids[owners[~shown]])
    feature_rows[node_set.name] = rows
    coarse = []
    for location in ds._coarse_scale_names():
        set_name = names[scale_of[location]]
        if set_name not in sets or set_name == node_set.name:
            continue
        coarse.append(set_name)
        entity = np.full(owners.size, -1, dtype=np.int64)
        entity[~shown] = ds.owner(location)[owners[~shown]]
        entity_ids = np.asarray(ds.entity_ids(location), dtype=np.int64)
        rows = np.full(owners.size, NO_ID, dtype=np.int64)
        known = entity >= 0
        rows[known] = _rows_of(sets[set_name].columns["id"], entity_ids[entity[known]])
        feature_rows[set_name] = rows
    if shown.any():                       # by their ids, up their complex links (display rows and vertices alike)
        ids = np.repeat(triangles.owner_ids, 3)[shown]
        unique, inverse = np.unique(ids, return_inverse=True)
        found = _rows_up(sets, unique, [node_set.name] + coarse)
        for set_name, rows in found.items():
            feature_rows[set_name][shown] = rows[inverse]
    positions = triangles.vertices.reshape(-1, 3)
    if not shared:
        return encoding3d.Mesh(name, positions=positions, feature_rows=feature_rows)
    positions, feature_rows, indices = _shared_vertices(positions, feature_rows)
    return encoding3d.Mesh(name, positions=positions, feature_rows=feature_rows, indices=indices)


def _shared_vertices(positions: np.ndarray, feature_rows: dict) -> tuple:
    """(positions, feature rows, indices): the vertices of equal position and rows once, in their first order."""
    keys = np.column_stack([np.ascontiguousarray(positions, dtype=np.float64).view(np.int64)]
                           + [np.asarray(rows, dtype=np.int64)[:, None] for rows in feature_rows.values()])
    keys = np.ascontiguousarray(keys)
    void = keys.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()
    _, first, inverse = np.unique(void, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")                 # the vertices in the order they first appear
    position_of = np.empty(order.size, dtype=np.int64)
    position_of[order] = np.arange(order.size)
    kept = first[order]
    return (positions[kept], {name: np.asarray(rows)[kept] for name, rows in feature_rows.items()},
            position_of[inverse.reshape(-1)])


def _rows_up(sets: dict, ids: np.ndarray, set_names: list) -> dict:
    """{set name: the row of each of *ids*' entity in that set (NO_ID: none)}: each id's own row, then its complex's,
    up the complex links (a display row's Compartment, its Cell, its Layer, its section...)."""
    table_ids, table_complex, table_set, table_row = [], [], [], []
    for k, set_name in enumerate(set_names):
        columns = sets[set_name].columns
        table_ids.append(np.asarray(columns["id"], dtype=np.int64))
        table_complex.append(np.asarray(columns["complex"], dtype=np.int64) if "complex" in columns
                             else np.full(columns["id"].size, NO_ID, dtype=np.int64))
        table_set.append(np.full(columns["id"].size, k, dtype=np.int64))
        table_row.append(np.arange(columns["id"].size, dtype=np.int64))
    every = np.concatenate(table_ids)
    order = np.argsort(every, kind="stable")
    every, complex_of = every[order], np.concatenate(table_complex)[order]
    set_of, row_of = np.concatenate(table_set)[order], np.concatenate(table_row)[order]
    out = {set_name: np.full(ids.size, NO_ID, dtype=np.int64) for set_name in set_names}
    current = np.asarray(ids, dtype=np.int64).copy()
    for _ in range(64):
        live = current != NO_ID
        if not live.any():
            break
        position = np.searchsorted(every, current).clip(0, every.size - 1)
        here = live & (every[position] == current)
        for k, set_name in enumerate(set_names):
            at = here & (set_of[position] == k) & (out[set_name] == NO_ID)
            out[set_name][at] = row_of[position[at]]
        current = np.where(here, complex_of[position], NO_ID)
    return out


# ── Grids → data structures ────────────────────────────────────────────────

def _grid_data_structure(identifier: str, ds, metadata: dict, standard_names: dict):
    encoding3d = _encoding3d()
    adaptive = isinstance(ds, AdaptiveGridDataStructure)
    extras = {"class": type(ds).__name__, "construction": ds._construction(), "variables": {},
              "aliases": ds.aliases()}
    count = ds.n_nodes()
    cells = {"id": np.asarray(ds.entity_ids("cell"), dtype=np.int64)}        # a regular grid's: the C-order index
    if adaptive:
        cells["level"] = np.asarray(ds._level, dtype=np.int64)
        cells["index"] = np.asarray(ds._index, dtype=np.int64)
    faces = {}
    edges = np.asarray(ds.edges(), dtype=np.int64).reshape(-1, 2)
    cell_ids = np.asarray(ds.entity_ids("cell"), dtype=np.int64)
    faces["source"], faces["target"] = cell_ids[edges[:, 0]], cell_ids[edges[:, 1]]
    scalars = {}
    for location, store in ds._var_stores().items():
        for name, values in store.items():
            values = np.asarray(values)
            target = {"cell": cells, "edge": faces, "scalar": scalars}[location]
            if location == "cell" and not adaptive:
                values = values.reshape((count,) + values.shape[len(ds.shape):])
            if location == "scalar":
                values = values.reshape((1,) + values.shape)
            column = f"metafspm_{name}" if name in ("id", "level", "index", "source", "target") else name
            target[column] = values
            extras["variables"][name] = dict(_json_meta(ds._variable_meta().get(name, {})), location=location,
                                             **({"column": column} if column != name else {}))
    cell_set = encoding3d.EntitySet("cell", columns=cells)
    face_relation = encoding3d.Relation("face", columns=faces, source_set="cell", target_set="cell")
    scalar_table = encoding3d.Table("scalars", columns=scalars) if scalars else None
    derived = set(ds.derived())
    for table in [cell_set, face_relation] + ([scalar_table] if scalar_table is not None else []):
        _declare(table, metadata, standard_names, derived)
    construction = ds._construction()
    grid = encoding3d.Grid(kind="adaptive" if adaptive else "regular", shape=tuple(construction["shape"]),
                        origin=tuple(construction["origin"]), spacing=tuple(construction["dx"]),
                        axes=tuple(ds.axes), periodic=tuple(bool(p) for p in construction["periodic"]),
                        z_up=False, max_level=int(construction.get("max_level", 0)), cells="cell")
    return encoding3d.DataStructure(identifier, kind="adaptive_grid" if adaptive else "regular_grid",
                                 entity_sets=[cell_set], relations=[face_relation] if edges.size else [],
                                 scalars=scalar_table, grid=grid, platform=PLATFORM, model=type(ds).__name__,
                                 extras={EXTRAS: extras})


# ── Frames ─────────────────────────────────────────────────────────────────

def to_frame(data_structures: dict, components: dict = None, mappings: dict = None, time: float = 0.,
             step: int = 0, time_unit="s", geometry: bool = True, standard_names: dict = None,
             scales=None, entities=None, variables=None, plant_names: dict = None, display_meshes: bool = True,
             display_plants=None, display: str = "instances", fold=None):
    """
    The encoding3d frame of *data_structures* ({id: DataStructure}; ids are identifiers), their variables declared
    by *components* ({id: [components]}), with *mappings* ({name: (CrossMapping, source id, target id)}), at *time*
    (in *time_unit*: "s", "min", "h", "d" or its number of seconds, as as_forcing's) and *step*. *geometry*: the
    MPGDataStructures' triangles as meshes; *display_meshes*: with them, the components' display_triangles() (shapes
    drawn for display only, e.g. GRANAP's sections; their data rows are written either way); *display_plants*: the
    plants whose display shapes are drawn, {id: [Plant vids or plant names (plant_names')]} or one list for every
    DataStructure (None: all; the others' reduced sections' interiors not recovered). *display*: how the display
    shapes draw the groups of a DataStructure's Reduction (any reduction, at any location):

      "instances"   (default) each reduced group an instance of its full template (an encoding3d instance set per
                    component giving display_instances: the template's shape and nodes once, a shared part; per
                    group its transform, owner, kept values; the unknown on the template's nodes a derived field,
                    its archetypes' recovery operators shared parts too, rebuilt by the reader for the instances it
                    selects: encoding3d.evaluate, ParaView's Select at scale); a DataStructure without such a
                    component draws its groups as with "full"
      "full"        the groups drawn whole in the display mesh, their interiors recovered as display rows
                    (metafspm_display_only): every value in the frame, the largest
      None          the reduced groups not drawn (the others are)

    *standard_names*: {metafspm name: standard name} on top of the vocabulary's metafspm column (a None value drops
    one). *fold*: ((x0, x1), (y0, y1)), the periodic pattern the plants' drawing is folded into, for display (a
    stand's leaves and roots beyond its pattern drawn where the periodic scene has them; fold_frame); None: as they
    are.

    For logging, *scales*, *entities* and *variables* filter what is written (see filter_frame), each a value for
    every DataStructure or a {id: value} dict; *plant_names*: {id: {plant name: Plant vid}}, to give *entities* by
    plant name. A filtered frame is a valid encoding3d file, but from_frame does not rebuild it.
    """
    if display not in DISPLAYS:
        raise ValueError(f"display: one of {DISPLAYS}, got {display!r}")
    encoding3d = _encoding3d()
    from openalea.encoding3d import vocabulary
    names = dict(vocabulary.platform_names("metafspm"), **(standard_names or {}))
    names = {key: value for key, value in names.items() if value}
    from openalea.metafspm.coupling.durations import duration, label
    frame = encoding3d.Frame(time=float(time), time_unit=label(duration(time_unit, "time_unit")), step=int(step),
                          writer={"platform": PLATFORM}, vocabulary_version=vocabulary.version())
    if variables is not None:
        variables = {identifier: resolved_variables(_for(variables, identifier), ds,
                                                    (components or {}).get(identifier, ()))
                     for identifier, ds in data_structures.items()}
    for identifier, ds in data_structures.items():
        metadata = _declarations((components or {}).get(identifier, ()))
        if isinstance(ds, MPGDataStructure):
            drawn = _display_plants_of(display_plants, identifier, (plant_names or {}).get(identifier, {}))
            frame.data_structures.append(_mpg_data_structure(identifier, ds, metadata, names, geometry,
                                                             (components or {}).get(identifier, ()), display_meshes,
                                                             drawn, display, _for(variables, identifier)))
        elif isinstance(ds, (ArrayDataStructure, AdaptiveGridDataStructure)):
            frame.data_structures.append(_grid_data_structure(identifier, ds, metadata, names))
        else:
            raise TypeError(f"'{identifier}': a {type(ds).__name__} cannot be exchanged (MPGDataStructure, "
                            "ArrayDataStructure and AdaptiveGridDataStructure can)")
    for name, (mapping, source, target) in (mappings or {}).items():
        rows, cells, weights = mapping.incidence()
        source_ds, target_ds = data_structures[source], data_structures[target]
        source_set = frame.data_structure(source)
        source_name = (source_set.entity_sets[0].name if not isinstance(source_ds, MPGDataStructure)
                       else _scale_names(source_ds.structure)[_location_scales(source_ds)["node"]])
        frame.mappings.append(encoding3d.Mapping(
            name, source=(source, source_name), target=(target, "cell"), method=getattr(mapping, "method", None),
            columns={"source": np.asarray(source_ds.entity_ids("node"), dtype=np.int64)[rows],
                     "target": np.asarray(target_ds.entity_ids("cell"), dtype=np.int64)[cells],
                     "weight": np.asarray(weights, dtype=np.float64)}))
    if fold is not None:
        fold_frame(frame, fold, [identifier for identifier, ds in data_structures.items()
                                 if isinstance(ds, MPGDataStructure)])
    if scales is not None or entities is not None or variables is not None:
        filter_frame(frame, scales=scales, entities=entities, variables=variables, plant_names=plant_names)
    return frame


def _folding(points: np.ndarray, pattern) -> np.ndarray:
    """The (n, 3) translations bringing *points* ((n, 3)) into the periodic *pattern* ((x0, x1), (y0, y1)): whole
    periods along x and y, none along z."""
    offset = np.zeros((len(points), 3))
    for axis, (low, high) in enumerate(pattern):
        period = float(high) - float(low)
        if period > 0:
            offset[:, axis] = -np.floor((points[:, axis] - float(low)) / period) * period
    return offset


def fold_frame(frame, pattern, identifiers=None):
    """
    Fold the plants' drawing of *frame* (changed in place, and returned) into the periodic *pattern* ((x0, x1),
    (y0, y1): the stand's, Scene.xrange and yrange), for display: the scene is periodic (its light and its soil), so a
    leaf or a root beyond the pattern is drawn where the pattern has it. Each mesh triangle moves by the whole periods
    that bring its centre into the pattern (a triangle stays whole; an indexed mesh's vertices move with their
    triangles), each segment (its ends' columns, the roots' tubes) by those of its middle, each instance by those of
    its placement. Of the graph data structures *identifiers* (None: every one); the frame marked as folded
    (from_frame does not rebuild it: the ends are no longer the plants').
    """
    from openalea.encoding3d import views
    pattern = tuple((float(low), float(high)) for low, high in pattern)
    for data_structure in frame.data_structures:
        if data_structure.kind != "graph" or (identifiers is not None and data_structure.id not in identifiers):
            continue
        for mesh in data_structure.meshes:
            if mesh.indices is None:
                triangles = mesh.positions.reshape(-1, 3, 3)
                offset = _folding(triangles.mean(axis=1), pattern)
                mesh.positions = (triangles + offset[:, None, :]).reshape(-1, 3)
            else:
                corners = mesh.indices.reshape(-1, 3)
                offset = _folding(mesh.positions[corners].mean(axis=1), pattern)
                moved = np.zeros_like(mesh.positions)
                moved[corners.reshape(-1)] = np.repeat(offset, 3, axis=0)
                mesh.positions = mesh.positions + moved
        for entity_set in data_structure.entity_sets:
            columns = views.segment_columns(entity_set)
            if not columns:
                continue
            start = np.stack([np.asarray(entity_set.columns[name], dtype=float) for name in columns[:3]], axis=1)
            end = np.stack([np.asarray(entity_set.columns[name], dtype=float) for name in columns[3:]], axis=1)
            offset = np.nan_to_num(_folding(0.5 * (start + end), pattern))
            for k, name in enumerate(columns):
                entity_set.columns[name] = np.asarray(entity_set.columns[name], dtype=float) + offset[:, k % 3]
        for instance_set in data_structure.instance_sets:
            transforms = np.asarray(instance_set.columns["transform"], dtype=float).reshape(-1, 4, 4)  # column major
            offset = _folding(transforms[:, 3, :3], pattern)
            transforms[:, 3, :3] += offset
            instance_set.columns["transform"] = transforms.reshape(-1, 16)
        data_structure.extras.setdefault(EXTRAS, {})["folded"] = [list(bounds) for bounds in pattern]
    return frame


def _display_plants_of(display_plants, identifier: str, names: dict):
    """The Plant vids of DataStructure *identifier* whose display shapes are drawn (None: all), from *display_plants*
    (to_frame's) and its plant names {name: vid}."""
    if display_plants is None:
        return None
    given = display_plants.get(identifier, None) if isinstance(display_plants, dict) else display_plants
    if given is None:
        return None
    vids = []
    for plant in ([given] if isinstance(given, (str, int, np.integer)) else list(given)):
        if isinstance(plant, str):
            if plant not in names:
                raise ValueError(f"display_plants: no plant '{plant}' in '{identifier}' ({sorted(names)})")
            plant = names[plant]
        vids.append(int(plant))
    return vids


def scene_to_frame(scene, geometry: bool = True, standard_names: dict = None, data_structures=None, scales=None,
                   entities=None, variables=None, display_meshes: bool = True, display_plants=None,
                   display: str = "instances", fold: bool = False):
    """The frame of a Scene now: each population's DataStructure (id: the population's name), each environment
    model's (id: the model's class name, numbered when it has several), the scene's CrossMappings.

    *geometry*, *display_meshes*, *display* and *display_plants* (the planting table's plant names, e.g. ("p0", "p3"),
    or a {population: names} dict): as in to_frame. *fold*: the plants' drawing folded into the scene's pattern
    ((0, xrange), (0, yrange): the stand's, periodic), for display (to_frame's fold). For logging: *data_structures*, the ids of those to write (default:
    all); *scales*, *entities* and *variables*, as in to_frame (filter_frame), *entities* also given by the planting
    table's plant names (e.g. "p0")."""
    found = scene.data_structures()
    written = {identifier: entry[0] for identifier, entry in found.items()}
    components = {identifier: entry[2] for identifier, entry in found.items()}
    identifier_of = {id(ds): identifier for identifier, ds in written.items()}
    mappings = {}
    for mapping in getattr(scene, "mappings", ()):
        if not hasattr(mapping, "incidence"):
            if type(mapping).__name__ != "UnionMapping":         # a union's elements are its parts': nothing to map
                warnings.warn(f"a {type(mapping).__name__} is not exchanged (CrossMappings are)")
            continue
        source, target = identifier_of.get(id(mapping.source)), identifier_of.get(id(mapping.target))
        if source is None or target is None:
            continue
        mappings[f"{source}_to_{target}"] = (mapping, source, target)
    if data_structures is not None:
        unknown = set(data_structures) - set(written)
        if unknown:
            raise ValueError(f"no data structure {sorted(unknown)} in the scene ({sorted(written)})")
        written = {identifier: ds for identifier, ds in written.items() if identifier in data_structures}
        mappings = {name: entry for name, entry in mappings.items() if entry[1] in written and entry[2] in written}
    plant_names = {population.name: {name: vid for vid, name in population.plant_names().items()}
                   for population in scene.populations if hasattr(population, "plant_names")}
    return to_frame(written, components=components, mappings=mappings, time=scene.time,
                    step=scene.iteration, time_unit="s", geometry=geometry, standard_names=standard_names,
                    scales=scales, entities=entities, variables=variables, plant_names=plant_names,
                    display_meshes=display_meshes, display_plants=display_plants, display=display,
                    fold=_scene_pattern(scene) if fold else None)


def _scene_pattern(scene):
    """The scene's periodic pattern ((0, xrange), (0, yrange)): the stand's, its soil's and its light's."""
    if getattr(scene, "xrange", None) is None or getattr(scene, "yrange", None) is None:
        raise ValueError("scene_to_frame(fold=True): the scene has no pattern (xrange, yrange)")
    return (0., float(scene.xrange)), (0., float(scene.yrange))


def export_frame(path: str, data_structures: dict, **options) -> None:
    """Write to_frame(data_structures, \\*\\*options) to the .glb file *path*."""
    _encoding3d().write(path, to_frame(data_structures, **options))


def export_scene(path: str, scene, **options) -> None:
    """Write scene_to_frame(scene, \\*\\*options) to the .glb file *path*."""
    _encoding3d().write(path, scene_to_frame(scene, **options))


# ── Views (encoding3d §8.1: the layers ParaView shows) ─────────────────────────

ROOT_LABELS = ("RootSegment", "RootApex")                       # the vocabulary's root SubOrgan labels
SHOOT_LABELS = ("LeafElement", "StemElement", "HiddenElement")  # its shoot ones
SHOOT_COLOR = [0.42, 0.68, 0.3]
SHOOT_COLORMAP = "Cool to Warm"                  # the default views' colour maps, one per part (ParaView presets)
ROOTS_COLORMAP = "Inferno (matplotlib)"
GRID_COLORMAP = "Viridis (matplotlib)"
SECTIONS_COLORMAP = "Blue Orange (divergent)"    # a reduction's instances, by the derived unknown


def _labels_for(data_structure, vocabulary_labels: tuple, others: list = ()) -> list:
    """The labels a data structure gives the vocabulary's *vocabulary_labels*: those, and its own names for them (its
    labels map), with *others* (more of its labels), sorted."""
    aliases = {platform for platform, standard in data_structure.labels.items() if standard in vocabulary_labels}
    return sorted(set(vocabulary_labels) | aliases | set(others))


def _shown_variable(declared: dict, entity_set, scale: str = None, location: str = None) -> str:
    """The variable a layer is coloured by: the first numeric state variable (not a descriptor) the components
    declare at *scale* (or *location*) that *entity_set* holds, else the first such input; None without one."""
    from openalea.metafspm.coupling.declaration import scale_names
    names = scale_names()
    for wanted in ("state_variable", "input_variable"):
        for name, metadata in declared.items():
            column = entity_set.columns.get(name)
            if (metadata.get("role") != wanted or metadata.get("kind") == "descriptor" or column is None
                    or column.dtype == object or column.dtype.kind not in "fiu"):
                continue
            at = metadata.get("location") if location is not None else names.get(metadata.get("scale"))
            if at == (location if location is not None else scale) or (at is None and location is None):
                return name
    return None


def _draws_triangles(ds) -> bool:
    try:
        ds.triangles()
    except (ValueError, NotImplementedError, AttributeError):
        return False
    return True


def _plant_layers(data_structure, ds, declared: dict) -> list:
    """A population's default layers: its shoot mesh coloured by an Organ variable, its root segments as tubes on
    radius coloured by a SubOrgan variable, each component's display mesh (e.g. sections) hidden."""
    from openalea.encoding3d import views
    sets = {entity_set.name: entity_set for entity_set in data_structure.entity_sets}
    elements = sets.get("SubOrgan")
    if elements is None:
        return []
    present = sorted({label for label in elements.columns["label"].tolist() if label is not None}) \
        if "label" in elements.columns else []
    roots = _labels_for(data_structure, ROOT_LABELS)
    shoots = _labels_for(data_structure, SHOOT_LABELS, [label for label in present if label not in roots])
    layers = []
    meshes = {mesh.name: mesh for mesh in data_structure.meshes}
    drawn = "triangles" in meshes or (ds is not None and _draws_triangles(ds))
    if drawn and (set(present) - set(roots) or not present):          # shoot elements, or no element yet
        organs = sets.get("Organ")
        variable = _shown_variable(declared, organs, scale="Organ") if organs is not None else None
        layer = {"name": f"{data_structure.id} shoot",
                 "source": {"data_structure": data_structure.id, "kind": "mesh", "mesh": "triangles"},
                 "select": {"entity_set": "SubOrgan", "where": {"label": shoots}}}
        layer.update({"color_by": {"entity_set": "Organ", "variable": variable}, "colormap": SHOOT_COLORMAP}
                     if variable else {"color_by": None, "solid_color": SHOOT_COLOR})
        layers.append(layer)
    if views.segment_columns(elements):
        variable = _shown_variable(declared, elements, scale="SubOrgan")
        layer = {"name": f"{data_structure.id} roots",
                 "source": {"data_structure": data_structure.id, "kind": "segments", "entity_set": "SubOrgan"},
                 "select": {"entity_set": "SubOrgan", "where": {"label": roots}},
                 "color_by": {"entity_set": "SubOrgan", "variable": variable} if variable else None}
        if variable:
            layer["colormap"] = ROOTS_COLORMAP
        if "radius" in elements.columns:
            layer["tube"] = {"radius": "radius", "scale": 1.}
        layers.append(layer)
    for name in meshes:
        if name != "triangles":                       # a component's shapes for display only, e.g. its sections
            layers.append({"name": f"{data_structure.id} {name}",
                           "source": {"data_structure": data_structure.id, "kind": "mesh", "mesh": name},
                           "color_by": None, "visible": False})
    for instance_set in data_structure.instance_sets:  # a reduction's groups drawn as instances (display="instances")
        layer = {"name": f"{data_structure.id} {instance_set.name} instances",
                 "source": {"data_structure": data_structure.id, "kind": "instances",
                            "instance_set": instance_set.name},
                 "color_by": None, "visible": False}
        if instance_set.derived:                      # coloured by the reduced system's unknown, once rebuilt
            field = instance_set.derived[0]
            layer.update(color_by={"entity_set": field.entity_set, "variable": field.name},
                         colormap=SECTIONS_COLORMAP)
        layers.append(layer)
    return layers


def default_views(scene, frame=None) -> dict:
    """
    The default views of a Scene's frames (encoding3d's views.json, §8.1), one configuration "default":

    - per population: "<population> shoot", its mesh (the geometry provider's triangles) where the SubOrgans are shoot
      elements, coloured by the first Organ variable its components declare (a state variable not a descriptor, else
      an input; a solid green without one); "<population> roots", its SubOrgans with segment ends where they are root
      elements, as tubes on radius, coloured by the first such SubOrgan variable; each part its own colour map (shoot
      SHOOT_COLORMAP, roots ROOTS_COLORMAP, grids GRID_COLORMAP); each display mesh of a component
      ("<population> <component>", e.g. sections), hidden; each instance set (a reduction's groups drawn as
      instances: "<population> <component> instances"), hidden, coloured by its first derived field (the reduced
      system's unknown, SECTIONS_COLORMAP: its colours once the reader rebuilds it, for the instances selected or
      every one);
    - per environment grid: "<model>", its cells coloured by the first cell variable its components declare, at
      opacity 0.3.

    *frame*: the scene's frame (default: scene_to_frame(scene)), whose data structures, meshes and labels the layers
    are of: a population gets its shoot layer when it draws triangles and the frame has shoot elements (or no
    element yet), its roots layer when its SubOrgans have segment ends. A part that appears later (a shoot from
    roots-only frames) gets no default layer: give SceneRecorder(frame_views=) a views dict or file then.
    """
    frame = frame if frame is not None else scene_to_frame(scene)
    found = scene.data_structures()
    populations = {population.name for population in scene.populations}
    layers = []
    for data_structure in frame.data_structures:
        ds, _, components = found.get(data_structure.id, (None, None, ()))
        declared = _declarations(components)
        if data_structure.id in populations and data_structure.kind == "graph":
            layers += _plant_layers(data_structure, ds, declared)
        elif data_structure.grid is not None and data_structure.id not in populations:
            cells = data_structure.entity_set(data_structure.grid.cells)
            variable = _shown_variable(declared, cells, location="cell")
            layers.append({"name": data_structure.id, "source": {"data_structure": data_structure.id, "kind": "grid"},
                           "color_by": {"entity_set": cells.name, "variable": variable} if variable else None,
                           "colormap": GRID_COLORMAP, "opacity": 0.3})
    return {"encoding3d_views": 1, "active": "default", "configurations": {"default": {"layers": layers}}}


def write_views(path: str, views: dict) -> None:
    """Write *views* (encoding3d's views.json, §8.1) to *path*, checked against encoding3d's schema."""
    _encoding3d()
    from openalea.encoding3d import views as encoding3d_views
    encoding3d_views.write_views(path, views)


# ── Logging: filtered frames ────────────────────────────────────────────────

def _for(value, identifier: str):
    """A filter's value for the data structure *identifier*: its entry when *value* is a dict, else *value*."""
    return value.get(identifier) if isinstance(value, dict) else value


def filter_frame(frame, scales=None, entities=None, variables=None, plant_names: dict = None):
    """
    Keep, for logging, part of *frame* (changed in place, and returned); each filter is a value for every data
    structure, or a {data structure id: value} dict (None: unfiltered)::

      entities:   ids (vids) of entities kept with everything they contain at finer scales, e.g. a plant, an axis,
                  or a segment with its cross-section's layers, cells and compartments; also plant names of the
                  planting table, through *plant_names* ({data structure id: {name: Plant vid}}). Their containing
                  entities are kept too, so that every complex link resolves. Graphs only.
      scales:     names of the entity sets kept (e.g. ["Plant", "SubOrgan"]); a complex link over a dropped scale
                  goes to the nearest kept one. Graphs only.
      variables:  names of the variables kept in every table, or a {table: names} dict, a table being an entity set,
                  a relation or the scalars (by name) and "*" every table it does not name (a table it does not
                  name, without "*": unfiltered), e.g. {"Compartment": ["psi", "tissue"], "SubOrgan": ["radius"],
                  "*": []}; to to_frame and scene_to_frame, names may be "recorded" (as a whole or a table's): the
                  variables the declarations record (their record=: state variables and inputs, not parameters,
                  settings or a grid's geometric descriptors; unrecorded()), an Excluding here. A filtered table always keeps its structure (encoding3d's reserved columns: ids, labels,
                  complex and parent links, edge types, a relation's ends, a grid cell's level and index), its
                  segment ends and the display rows' marker (metafspm_display_only, e.g. a reduced section's
                  interiors); metafspm's own links (metafspm_complex, ...: from_frame's) only when named. The
                  instances' templates (shared parts) keep the columns their node set's filter keeps when the frame
                  is made (to_frame's variables): filter_frame leaves instance sets' columns as they are.

    Parents outside what is kept become none, relations and mappings keep the pairs whose ends are kept, meshes the
    triangles of kept entities, instance sets (display="instances") the instances of kept owners (none when their
    owners' scale is dropped). The result is a valid encoding3d frame, marked as filtered: from_frame does not rebuild it.
    filtered_view_columns checks views against a variables filter.
    """
    for data_structure in frame.data_structures:
        wanted = {"entities": _for(entities, data_structure.id), "scales": _for(scales, data_structure.id),
                  "variables": _for(variables, data_structure.id)}
        if all(value is None for value in wanted.values()):
            continue
        graph = data_structure.kind == "graph"
        if not graph and (wanted["entities"] is not None or wanted["scales"] is not None):
            raise ValueError(f"'{data_structure.id}' is a {data_structure.kind}: only its variables can be filtered")
        if wanted["entities"] is not None:
            names = (plant_names or {}).get(data_structure.id, {})
            ids = [names[item] if isinstance(item, str) else int(item) for item in np.atleast_1d(wanted["entities"])]
            _keep_entities(data_structure, ids)
        if wanted["scales"] is not None:
            _keep_scales(data_structure, list(wanted["scales"]))
        if wanted["variables"] is not None:
            _keep_variables(data_structure, wanted["variables"])
        data_structure.extras.setdefault(EXTRAS, {})["filtered"] = {
            key: (None if value is None else _filter_described(value)) for key, value in wanted.items()}
    _filter_mappings(frame)
    return frame


def _filter_described(value):
    """A filter's value as the frame's extras note it: a list (names or ids), {table: names} for variables, and
    {"excluding": names} for a value given as "recorded" (every variable but those)."""
    if isinstance(value, Excluding):
        return {"excluding": sorted(value.names)}
    if isinstance(value, dict):
        return {str(table): _filter_described(names) if isinstance(names, Excluding)
                else sorted(_table_variables(names, table)) for table, names in value.items()}
    return [item if isinstance(item, str) else int(item) for item in np.atleast_1d(value)]


def _table_variables(wanted, table: str):
    """The variables a variables filter (*wanted*: names, or {table: names, "*": names}) keeps in *table* (a set,
    or an Excluding: every variable but some), or None: *table* unfiltered."""
    names = wanted.get(table, wanted.get("*")) if isinstance(wanted, dict) else wanted
    if names is None or isinstance(names, Excluding):
        return names
    if isinstance(names, str) and names == RECORDED:
        raise ValueError("a variables filter 'recorded' is resolved from the declarations: give it to to_frame or "
                         "scene_to_frame (resolved_variables), not to filter_frame")
    return {str(name) for name in ([names] if isinstance(names, str) else names)}


def view_columns(views: dict) -> dict:
    """{data structure id: {table: columns}} that *views* (encoding3d's views.json) select, colour or size tubes
    by: the columns a frame keeps whatever its variables filter (with_view_columns)."""
    from openalea.encoding3d import model
    from openalea.encoding3d.views import SEGMENT_ENDS
    structure = set(model.RESERVED["entity_set"]) | {DISPLAY_ONLY} | {name for pair in SEGMENT_ENDS for name in pair}
    found = {}
    for content in (views or {}).get("configurations", {}).values():
        for layer in content.get("layers", ()):
            source = layer["source"]
            select, color_by = layer.get("select") or {}, layer.get("color_by") or {}
            used = [(select.get("entity_set"), column) for column in select.get("where", {})]
            used.append((color_by.get("entity_set"), color_by.get("variable")))
            used.append((source.get("entity_set"), (layer.get("tube") or {}).get("radius")))
            for table, column in used:
                if table and column and column not in structure:
                    found.setdefault(source["data_structure"], {}).setdefault(table, set()).add(column)
    return found


def with_view_columns(variables, views: dict, identifiers):
    """A variables filter (filter_frame's: a value for every data structure, or {id: value}) that also keeps the
    columns *views* show (view_columns), for the data structures *identifiers*: a table filtered by names gets
    them added, one filtered by an Excluding (\"recorded\") has them no longer excluded; a table unfiltered stays so."""
    if variables is None:
        return None
    shown = view_columns(views)
    out = {}
    for identifier in identifiers:
        wanted = _for(variables, identifier)
        columns = shown.get(identifier)
        if wanted is None or not columns:
            out[identifier] = wanted
            continue
        if not isinstance(wanted, dict):
            wanted = {"*": wanted}
        wanted = dict(wanted)
        for table, names in columns.items():
            kept = _table_variables(wanted, table)
            if kept is None:
                continue
            wanted[table] = kept | names if isinstance(kept, Excluding) else sorted(kept | names)
        out[identifier] = wanted
    return out


def _always_kept(table) -> set:
    """The columns a variables filter keeps in *table*: its structure, its segment ends, the display rows' marker."""
    from openalea.encoding3d import views
    kept = set(table.reserved) | {DISPLAY_ONLY}
    if table.table_kind == "entity_set":
        kept.update(views.segment_columns(table))
    return kept


def filtered_view_columns(views: dict, variables) -> list:
    """The columns *views* (encoding3d's views.json) select, colour or size tubes by that a variables filter (a
    value for every data structure, or a {data structure id: value} dict, as filter_frame's) drops, as
    "configuration/layer: data structure/entity set/column" lines ([]: none)."""
    from openalea.encoding3d import model
    from openalea.encoding3d.views import SEGMENT_ENDS
    structure = set(model.RESERVED["entity_set"]) | {DISPLAY_ONLY} | {name for pair in SEGMENT_ENDS for name in pair}
    dropped = []
    for configuration, content in views.get("configurations", {}).items():
        for layer in content.get("layers", ()):
            source = layer["source"]
            wanted = _for(variables, source["data_structure"])
            if wanted is None:
                continue
            select, color_by = layer.get("select") or {}, layer.get("color_by") or {}
            used = [(select.get("entity_set"), column) for column in select.get("where", {})]
            used.append((color_by.get("entity_set"), color_by.get("variable")))
            used.append((source.get("entity_set"), (layer.get("tube") or {}).get("radius")))
            for table, column in used:
                if not column or column in structure:
                    continue
                kept = _table_variables(wanted, table)
                if kept is not None and column not in kept:
                    dropped.append(f"{configuration}/{layer['name']}: {source['data_structure']}/{table}/{column}")
    return dropped


def _complex_links(data_structure) -> dict:
    """{id: the id of its complex} over every entity set (-1: none)."""
    links = {}
    for entity_set in data_structure.entity_sets:
        if "complex" in entity_set.columns:
            links.update(zip(entity_set.columns["id"].tolist(), entity_set.columns["complex"].tolist()))
    return links


def _select_rows(table, rows: np.ndarray) -> None:
    for name in list(table.columns):
        table.columns[name] = table.columns[name][rows]


def _keep_entities(data_structure, ids: list) -> None:
    links = _complex_links(data_structure)
    known = {identifier for entity_set in data_structure.entity_sets for identifier in entity_set.ids().tolist()}
    missing = sorted(set(ids) - known)
    if missing:
        raise ValueError(f"'{data_structure.id}' has no entity {missing[:5]}")
    kept = set(ids)
    for identifier in ids:                                     # their containing entities, up to the coarsest scale
        while links.get(identifier, NO_ID) != NO_ID:
            identifier = links[identifier]
            kept.add(identifier)
    within = set(ids)
    for entity_set in sorted(data_structure.entity_sets, key=lambda entity_set: entity_set.rank or 0):
        if "complex" in entity_set.columns:                    # everything they contain, scale by scale
            inside = np.isin(entity_set.columns["complex"], list(within))
            within.update(entity_set.columns["id"][inside].tolist())
    kept |= within
    old_rows = {}
    for entity_set in data_structure.entity_sets:
        keep = np.isin(entity_set.columns["id"], list(kept))
        old_rows[entity_set.name] = np.flatnonzero(keep)
        _select_rows(entity_set, keep)
        if "parent" in entity_set.columns:
            outside = ~np.isin(entity_set.columns["parent"], list(kept))
            entity_set.columns["parent"] = np.where(outside, NO_ID, entity_set.columns["parent"])
            entity_set.columns["edge_type"] = np.where(outside, "none", entity_set.columns["edge_type"]).astype(object)
    for relation in data_structure.relations:
        _select_rows(relation, np.isin(relation.columns["source"], list(kept))
                     & np.isin(relation.columns["target"], list(kept)))
    data_structure.meshes = [mesh for mesh in (_mesh_rows(mesh, old_rows) for mesh in data_structure.meshes)
                             if mesh is not None]
    for instance_set in data_structure.instance_sets:        # the instances of the kept owners
        _select_instances(instance_set, np.isin(instance_set.columns["owner"], list(kept))
                          if "owner" in instance_set.columns else np.ones(instance_set.count, dtype=bool))
    data_structure.instance_sets = [instance_set for instance_set in data_structure.instance_sets
                                    if instance_set.count]


def _select_instances(instance_set, keep: np.ndarray) -> None:
    """*instance_set* reduced to its instances *keep* (a mask): its rows, vectors and derived fields' terms."""
    rows = np.flatnonzero(keep)
    _select_rows(instance_set, rows)
    instance_set.vectors = {name: vector.take(rows) for name, vector in instance_set.vectors.items()}
    for field in instance_set.derived:
        field.operators, field.weights = field.operators[rows], field.weights[rows]


def _mesh_rows(mesh, old_rows: dict):
    """*mesh* with the triangles of the kept entities, its feature rows renumbered (None: no triangle left)."""
    encoding3d = _encoding3d()
    renumber = {}
    for name, rows in old_rows.items():
        if name in mesh.feature_rows:
            new = np.full(max(rows.max() + 1 if rows.size else 0, mesh.feature_rows[name].max() + 1), NO_ID)
            new[rows] = np.arange(rows.size)
            renumber[name] = new
    finest = next(name for name in reversed(list(mesh.feature_rows)) if name in renumber) if renumber else None
    if finest is None:
        return mesh
    if mesh.indices is not None:                       # the kept triangles, then the vertices they use
        triangles = mesh.indices.reshape(-1, 3)
        corners = mesh.feature_rows[finest][triangles[:, 0]]
        kept_triangles = (corners >= 0) & (renumber[finest][np.maximum(corners, 0)] >= 0)
        if not kept_triangles.any():
            return None
        used, indices = np.unique(triangles[kept_triangles], return_inverse=True)
        rows = {name: np.where(values[used] >= 0, renumber[name][np.maximum(values[used], 0)], NO_ID)
                if name in renumber else values[used] for name, values in mesh.feature_rows.items()}
        return encoding3d.Mesh(mesh.name, positions=mesh.positions[used], feature_rows=rows,
                               indices=indices.reshape(-1), display={name: values[used] for name, values in
                                                                     mesh.display.items()},
                               display_sources=dict(mesh.display_sources))
    corners = mesh.feature_rows[finest].reshape(-1, 3)[:, 0]
    keep = np.repeat((corners >= 0) & (renumber[finest][np.maximum(corners, 0)] >= 0), 3)
    if not keep.any():
        return None
    rows = {name: np.where(values[keep] >= 0, renumber[name][np.maximum(values[keep], 0)], NO_ID)
            if name in renumber else values[keep] for name, values in mesh.feature_rows.items()}
    return encoding3d.Mesh(mesh.name, positions=mesh.positions[keep], feature_rows=rows,
                        display={name: values[keep] for name, values in mesh.display.items()},
                        display_sources=dict(mesh.display_sources))


def _keep_scales(data_structure, names: list) -> None:
    unknown = set(names) - {entity_set.name for entity_set in data_structure.entity_sets}
    if unknown:
        raise ValueError(f"'{data_structure.id}' has no scale {sorted(unknown)} "
                         f"({[entity_set.name for entity_set in data_structure.entity_sets]})")
    links = _complex_links(data_structure)
    kept = [entity_set for entity_set in data_structure.entity_sets if entity_set.name in names]
    kept_ids = {identifier for entity_set in kept for identifier in entity_set.ids().tolist()}
    for entity_set in kept:
        if "complex" not in entity_set.columns:
            continue
        complexes = entity_set.columns["complex"].tolist()
        for row, identifier in enumerate(complexes):            # over the dropped scales, to the nearest kept one
            while identifier != NO_ID and identifier not in kept_ids:
                identifier = links.get(identifier, NO_ID)
            complexes[row] = identifier
        if kept.index(entity_set) == 0:
            del entity_set.columns["complex"]                   # now the coarsest scale kept
        else:
            entity_set.columns["complex"] = np.asarray(complexes, dtype=np.int64)
    data_structure.entity_sets = kept
    data_structure.relations = [relation for relation in data_structure.relations
                                if relation.source_set in names and relation.target_set in names]
    for mesh in data_structure.meshes:
        mesh.feature_rows = {name: rows for name, rows in mesh.feature_rows.items() if name in names}
        mesh.display_sources = {name: source for name, source in mesh.display_sources.items() if source in names}
    data_structure.meshes = [mesh for mesh in data_structure.meshes if mesh.feature_rows]
    data_structure.instance_sets = [instance_set for instance_set in data_structure.instance_sets
                                    if instance_set.owners is None or instance_set.owners in names]


def _keep_variables(data_structure, wanted) -> None:
    tables = list(data_structure.entity_sets) + list(data_structure.relations)
    for table in tables + ([data_structure.scalars] if data_structure.scalars is not None else []):
        names = _table_variables(wanted, table.name)
        if names is None:
            continue
        kept = names | _always_kept(table)
        for column in [column for column in table.columns if column not in kept]:
            del table.columns[column]
            table.variables.pop(column, None)
    if data_structure.scalars is not None and not data_structure.scalars.columns:
        data_structure.scalars = None


def _filter_mappings(frame) -> None:
    by_id = {data_structure.id: data_structure for data_structure in frame.data_structures}
    kept = []
    for mapping in frame.mappings:
        ends = []
        for data_structure_id, set_name in (mapping.source, mapping.target):
            data_structure = by_id.get(data_structure_id)
            sets = {entity_set.name: entity_set for entity_set in data_structure.entity_sets} if data_structure else {}
            ends.append(sets.get(set_name))
        if None in ends:
            continue
        _select_rows(mapping, np.isin(mapping.columns["source"], ends[0].ids())
                     & np.isin(mapping.columns["target"], ends[1].ids()))
        kept.append(mapping)
    frame.mappings = kept


# ── Data structures from a frame ────────────────────────────────────────────

def from_frame(frame) -> dict:
    """{id: DataStructure} of the data structures metafspm wrote in *frame* (their extras rebuild them exactly).
    Data structures from other platforms are left out with a warning; mappings are left out (scenes rebuild them)."""
    out = {}
    for data_structure in frame.data_structures:
        extras = data_structure.extras.get(EXTRAS)
        if extras is None:
            warnings.warn(f"'{data_structure.id}' was not written by metafspm: not rebuilt")
            continue
        if "filtered" in extras:
            raise ValueError(f"'{data_structure.id}' was written filtered, for logging ({extras['filtered']}): "
                             "it cannot be rebuilt (write it unfiltered to exchange it)")
        if "folded" in extras:
            raise ValueError(f"'{data_structure.id}' was written folded into its pattern ({extras['folded']}), for "
                             "display: it cannot be rebuilt (write it unfolded to exchange it)")
        if extras["class"] == "MPGDataStructure":
            if "absent_rows" not in extras:
                raise ValueError(f"'{data_structure.id}' was written by a metafspm before 2026-11-09 (its properties' "
                                 "absent vids listed one by one): rewrite the frame to rebuild it")
            out[data_structure.id] = _mpg_from(data_structure, extras)
        else:
            out[data_structure.id] = _grid_from(data_structure, extras)
    return out


def import_frame(path: str) -> dict:
    """from_frame of the .glb file *path*."""
    return from_frame(_encoding3d().read(path))


def _mpg_from(data_structure, extras: dict):
    g = MPG()
    if sorted(set(_anchor_vids(g))) != extras["anchors"]:
        raise ValueError(f"'{data_structure.id}': the MPG's scale anchors differ from the written ones")
    scales = extras["scales"]
    tables = {table.name: _without_display_rows(table) for table in data_structure.entity_sets + data_structure.relations}
    # The vertices and their own links, in vid order (the order of the MPG's component and children lists)
    links = []
    for name, table in tables.items():
        scale = scales[name]
        columns = table.columns
        links += [(int(v), scale, int(c), int(p)) for v, c, p in
                  zip(columns["id"], columns["metafspm_complex"], columns["metafspm_parent"])]
    for vid, scale, complex_, parent in sorted(links):
        g._scale[vid] = scale
        if complex_ >= 0:
            g._complex[vid] = complex_
            g._components.setdefault(complex_, []).append(vid)
        if parent >= 0:
            g._parent[vid] = parent
            g._children.setdefault(parent, []).append(vid)
    g._id = extras["last_id"]
    # The properties
    labels, edge_types = [], {}
    for name, table in tables.items():
        codes = extras["label_codes"].get(name, {})
        column = "relation" if isinstance(table, _encoding3d().Relation) else "label"
        if column in table.columns:
            named = table.columns[column] != "_none"
            labels.append((table.columns["id"][named],
                           np.array([codes[label] for label in table.columns[column][named]], dtype=np.float64)))
        edges = table.columns["metafspm_edge_type"]
        edge_types.update(zip(table.columns["id"][edges > 0].tolist(),
                              [EDGE_CODES[int(code)] for code in edges[edges > 0]]))
    _merge_arrays(g, "label", labels)
    _merge(g, "edge_type", "dict", edge_types)
    for property_name, entry in extras["properties"].items():
        arrays, values = [], {}
        for name, kind in entry["dtypes"].items():
            table = tables[name]
            ids = table.columns["id"]
            absent = _ranges_mask(extras["absent_rows"].get(f"{name}.{property_name}", ()), ids.size)
            none = _ranges_mask(extras["none_rows"].get(f"{name}.{property_name}", ()), ids.size)
            column = table.columns.get(entry["column"]) if kind is not None else None
            if entry["container"] == "ArrayDict":
                kept = ~absent & ~none
                arrays.append((ids[kept], np.asarray(column, dtype=np.float64)[kept]))
                continue
            convert = {"float64": float, "int64": int, "bool": bool, "json": _decoded}.get(kind, str)
            for vid, here, missing, value in zip(ids.tolist(), ~absent, none,
                                                 column if column is not None else [None] * ids.size):
                if here:
                    values[vid] = None if missing or column is None else convert(value)
        if entry["container"] == "ArrayDict":
            _merge_arrays(g, property_name, arrays)
        else:
            _merge(g, property_name, "dict", values)
    for name, table in tables.items():
        if isinstance(table, _encoding3d().Relation):
            _merge_arrays(g, "n_id_a", [(table.columns["id"], table.columns["source"].astype(np.float64))])
            _merge_arrays(g, "n_id_b", [(table.columns["id"], table.columns["target"].astype(np.float64))])
    for property_name, entry in extras.get("anchor_properties", {}).items():
        prop = g._properties.get(property_name)
        if prop is None:
            prop = g._properties[property_name] = ArrayDict(dtype=np.float64 if entry["container"] == "ArrayDict"
                                                            else object)
        for vid, text in entry["values"].items():
            if text is None:
                prop.pop(int(vid), None)
            else:
                prop[int(vid)] = _decoded(text)
    for property_name, container in extras.get("empty_properties", {}).items():
        g._properties.setdefault(property_name, ArrayDict() if container == "ArrayDict" else {})
    for property_name in extras.get("missing_properties", ()):
        g._properties.pop(property_name, None)
    ds = MPGDataStructure._from_construction(dict(extras["construction"]), {"mtg": g, "state": {"_wiring": None}})
    _restore_variables(ds, data_structure, extras, tables)
    for name in extras["tracked"]:
        meta = ds._variable_meta()[name]
        ds._track_mtg(VariableSpec(name=name, location=extras["variables"][name]["location"], scale=meta.get("scale"),
                                  aggregation=meta.get("aggregation"), weight=meta.get("weight"), kind=meta.get("kind"),
                                  role=meta.get("role"), default=meta.get("default", 0.),
                                  on_grow=meta.get("on_grow", "default"), dtype=meta.get("dtype", float),
                                  shape=tuple(meta.get("shape", ()))))
        ds._mark_mtg_synced(name)
    return ds


class _Rows:
    """A table's rows but its display rows (a reduced section's interior, for display only: not MPG vertices)."""

    def __init__(self, table, keep):
        self.name, self.variables = table.name, getattr(table, "variables", {})
        self.columns = {name: np.asarray(column)[keep] for name, column in table.columns.items()
                        if name != DISPLAY_ONLY}


def _without_display_rows(table):
    if DISPLAY_ONLY not in table.columns:
        return table
    return _Rows(table, ~np.asarray(table.columns[DISPLAY_ONLY], dtype=bool))


def _anchor_vids(g) -> list:
    return [g.root] + [int(v) for v in np.flatnonzero(g.topology_arrays()["is_anchor"])]


def _merge_arrays(g, name: str, parts: list) -> None:
    """Add [(vids, values)] to the MPG's ArrayDict property *name* (created when missing), at once."""
    existing = g._properties.get(name)
    keys, values = [], []
    if existing is not None and len(existing):       # a new MPG's column (its anchors' values), or none
        existing_keys, existing_values = existing.items_arrays()
        keys, values = [np.asarray(existing_keys)], [np.asarray(existing_values, dtype=np.float64)]
    keys += [np.asarray(vids, dtype=np.int64) for vids, _ in parts]
    values += [np.asarray(part, dtype=np.float64) for _, part in parts]
    g._properties[name] = _arraydict(np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64),
                                     np.concatenate(values) if values else np.zeros(0))


def _merge(g, name: str, container: str, values: dict) -> None:
    """Add *values* ({vid: value}) to the MPG's property *name* (created as a float column, "ArrayDict", or an object
    column, "dict")."""
    existing = g._properties.get(name)
    if container == "ArrayDict":
        keys = list(existing.keys()) if existing is not None else []
        old = [existing[k] for k in keys] if existing is not None else []
        new_keys = [k for k in values if values[k] is not None]
        g._properties[name] = ArrayDict.from_items(np.asarray(keys + new_keys, dtype=np.int64),
                                                   np.asarray(old + [values[k] for k in new_keys], dtype=np.float64))
    else:
        merged = dict(zip(*(existing.items_arrays()[0].tolist(), existing.items_arrays()[1].tolist()))) \
            if existing is not None else {}
        merged.update(values)
        g._properties[name] = ArrayDict.from_items(list(merged), list(merged.values()), dtype=object)


def _restore_variables(ds, data_structure, extras: dict, tables: dict) -> None:
    stores = ds._var_stores()
    for store in stores.values():
        store.clear()
    scale_names = {scale: name for name, scale in extras.get("scales", {}).items()}
    scale_of = _location_scales(ds) if isinstance(ds, MPGDataStructure) else {}
    for name, meta in extras["variables"].items():
        location, column = meta["location"], meta.get("column", name)
        if location == "scalar":
            values = data_structure.scalars.columns[column][0]
        elif isinstance(ds, MPGDataStructure):
            table = tables[scale_names[scale_of[location]]]
            ids = np.asarray(ds.entity_ids(location), dtype=np.int64)
            rows = _edge_rows(table.columns, ids) if location == "edge" else _rows_of(table.columns["id"], ids)
            values = table.columns[column][rows]
        else:
            table = data_structure.entity_sets[0] if location == "cell" else data_structure.relations[0]
            values = table.columns[column]
            if location == "cell" and isinstance(ds, ArrayDataStructure):
                values = values.reshape(tuple(ds.shape) + values.shape[1:])
        restored = _restored_meta(meta)
        dtype = restored.get("dtype", float)
        if dtype is object:
            values = np.asarray(values, dtype=object)
            array = np.empty(values.shape, dtype=object)
            for index, value in np.ndenumerate(values):
                array[index] = _decoded(value) if meta.get("encoding") == "json" else value
        else:
            array = np.array(values, dtype=np.int64 if dtype is int else np.float64)
        stores[location][name] = array
        ds._variable_meta()[name] = restored
    if extras.get("aliases"):
        ds.__dict__["_aliases"] = dict(extras["aliases"])


def _grid_from(data_structure, extras: dict):
    construction = dict(extras["construction"])
    if extras["class"] == "AdaptiveGridDataStructure":
        cells = data_structure.entity_sets[0].columns
        ds = AdaptiveGridDataStructure._from_construction(construction, {"state": {"_level": cells["level"],
                                                                                   "_index": cells["index"]}})
    else:
        ds = ArrayDataStructure._from_construction(construction, {})
    _restore_variables(ds, data_structure, extras, {})
    return ds
