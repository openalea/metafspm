"""
metafspm's DataStructures as fspm-TF frames (the cross-platform scene format, https://github.com/openalea/fspm-TF),
and back.

    frame = to_frame({"plants": ds, "soil": grid}, components={"plants": [growth, carbon]},
                     mappings={"roots_in_soil": (cross_mapping, "plants", "soil")})
    frame = scene_to_frame(scene)                  every population's and environment model's DataStructure, and the
                                                   scene's mappings
    export_frame(path, ...), export_scene(path, scene)       the same, written to a .glb file
    data_structures = from_frame(frame)            {id: DataStructure}, rebuilt: import_frame(path) from a file

What a frame holds:

  - an MPGDataStructure: one entity set per MPG scale holding vertices (Plant, Axis, ... SubOrgan, Layer, Cell,
    Compartment; ids are the vids), the Connections as a relation (n_id_a → n_id_b); as columns, every MPG property
    and every variable of the DataStructure at that scale's location; its scalars; the triangles of its geometry
    provider as a mesh, by node and by coarser scale, and a mesh for each component drawing shapes for display only
    (display_triangles(), e.g. GRANAP's sections). The MPG's own links (the raw complex and parent of each
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

_KINDS = {"intensive": "intensive", "extensive": "extensive", "massic_concentration": "massic_concentration",
          "volumic_concentration": "volumic_concentration", "descriptor": "descriptor",
          "NonInertialIntensive": "intensive", "NonInertialExtensive": "extensive"}
_ROLES = {"state_variable": "state", "plant_scale_state": "state", "input": "input", "parameter": "parameter",
          "setting": "setting", "forcing": "input"}
_DIMENSIONLESS = {"", "adim", "-", "1", "dimensionless", "adimensional", "unitless"}
_SYMBOLS = {"°C": "Cel", "°": "deg", "day": "d", "days": "d", "yr": "a", "°Cd": "K.d", "°C.d": "K.d"}
_GRAPH_RESERVED = ("id", "label", "complex", "parent", "edge_type", "level", "index")
_RELATION_RESERVED = ("id", "source", "target", "relation")


def _fspm_tf():
    try:
        from openalea import fspm_tf
    except ImportError:
        raise ImportError("openalea.metafspm.io.fspm_tf needs the openalea.fspm_tf package (clone fspm-TF next to "
                          "metafspm, then pip install -e ../fspm-TF)") from None
    return fspm_tf


# ── Units, standard names, declarations ─────────────────────────────────────

def ucum_unit(unit) -> str:
    """The UCUM code of a metafspm declaration's unit: 'mm3 s-1 MPa-1' → 'mm3.s-1.MPa-1', '°C' → 'Cel', 'adim' →
    '1', 'mol of hexose' → 'mol{hexose}'; a unit with no UCUM code becomes an annotation ('{phyllochron}')."""
    from openalea.fspm_tf import ucum
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


def declarations(components) -> dict:
    """{variable name: declaration metadata} of the variables the *components* declare (the first declaring each)."""
    found = {}
    for component in components or ():
        if not is_dataclass(component):
            continue
        for item in fields(type(component)):
            if "variable_type" in item.metadata or "unit" in item.metadata:
                found.setdefault(item.name, dict(item.metadata))
    return found


def _variable(name: str, metadata: dict, standard_names: dict, derived: bool = False):
    fspm_tf = _fspm_tf()
    from openalea.fspm_tf import ucum, vocabulary
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

    def number(value):
        return float(value) if isinstance(value, (int, float, np.number)) and np.isfinite(value) else None

    references = metadata.get("references")
    references = (references,) if isinstance(references, str) and references.strip() else tuple(references or ())
    doi = tuple(str(item) for item in (metadata.get("DOI") or ()) if str(item).strip())
    return fspm_tf.Variable(unit=unit, standard_name=standard_name, kind=_KINDS.get(metadata.get("state_variable_type")),
                            role="derived" if derived else _ROLES.get(metadata.get("variable_type"), "state"),
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


def _jsonable(value):
    """*value* as JSON values, tuples, arrays, non-text keys and non-finite floats marked, so that _decoded gives it
    back; TypeError for what JSON cannot hold."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__float__": repr(value)}
    if isinstance(value, tuple):
        return {"__tuple__": [_jsonable(item) for item in value]}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"__array__": [_jsonable(item) for item in value.tolist()], "dtype": str(value.dtype)}
    if isinstance(value, dict):
        if all(isinstance(key, str) and not key.startswith("__") for key in value):
            return {key: _jsonable(item) for key, item in value.items()}
        return {"__dict__": [[_jsonable(key), _jsonable(item)] for key, item in value.items()]}
    raise TypeError(f"a {type(value).__name__} cannot be written as JSON")


def _encoded(value) -> str:
    return json.dumps(_jsonable(value), separators=(",", ":"), allow_nan=False)


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
        return item
    return json.loads(text, object_hook=hook)


def _hashable(key):
    return tuple(_hashable(item) for item in key) if isinstance(key, list) else key


_MISSING = object()


def _lookup(prop, vids: np.ndarray) -> tuple:
    """(present, values) of the MPG property *prop* at *vids*: a mask, and the values (a float array, NaN where
    absent, for an ArrayDict, read in bulk from its sorted keys; a list, None where absent, for a dict)."""
    if isinstance(prop, ArrayDict):
        keys, stored = prop.order[:prop.size], prop.arr[:prop.size]
        if keys.size == 0:
            return np.zeros(vids.size, dtype=bool), np.full(vids.size, np.nan)
        position = np.clip(np.searchsorted(keys, vids), 0, keys.size - 1)
        present = keys[position] == vids
        return present, np.where(present, stored[position], np.nan)
    get = prop.get
    values = [get(v, _MISSING) for v in vids.tolist()]
    present = np.array([value is not _MISSING for value in values], dtype=bool)
    return present, [None if value is _MISSING else value for value in values]


def _arraydict(keys: np.ndarray, values: np.ndarray) -> ArrayDict:
    """An ArrayDict of unique *keys* and their *values*, built from the arrays at once."""
    keys, values = np.asarray(keys, dtype=np.int64), np.asarray(values, dtype=np.float64)
    order = np.argsort(keys, kind="stable")
    out = ArrayDict(init_capacity=keys.size)
    out.order[:keys.size], out.arr[:keys.size] = keys[order], values[order]
    out.size = int(keys.size)
    out.vid2idx = dict(zip(out.order[:out.size].tolist(), range(out.size)))
    return out


# ── MPGDataStructure → data structure ───────────────────────────────────────

def _scale_names(g) -> dict:
    return {value: name for name, value in vars(type(g.scales)).items()
            if isinstance(value, int) and not name.startswith("_")}


def _location_scales(ds) -> dict:
    """{DataStructure location: MPG scale} of a graph DataStructure's locations other than scalar."""
    g = ds.structure
    scales = {"node": g.scales.Compartment if ds._anatomy else ds._from_scale, "edge": g.scales.Connection}
    for name in ds._coarse_scale_names():
        scales[name] = getattr(g.scales, name)
    return scales


def _mpg_data_structure(identifier: str, ds, metadata: dict, standard_names: dict, geometry: bool, components=()):
    fspm_tf = _fspm_tf()
    ds.flush_mtg()                     # the MPG holds the variables' current values, as ds.mtg gives it
    g = ds.structure
    names = _scale_names(g)
    arrays = g.topology_arrays()
    anchors = set(np.flatnonzero(arrays["is_anchor"]).tolist()) | {g.root}
    by_scale = {}
    for vid in sorted(g._scale):
        if vid not in anchors:
            by_scale.setdefault(g._scale[vid], []).append(vid)
    connection = g.scales.Connection
    translator = dict(g.labels.translator)
    properties = g.properties()                    # read only: g.property() would create a missing property
    label_property, edge_property = properties.get("label", {}), properties.get("edge_type", {})
    extras = {"class": "MPGDataStructure", "construction": ds._construction(), "last_id": int(g._id),
              "anchors": sorted(int(v) for v in anchors), "scales": {}, "label_codes": {}, "properties": {},
              "absent": {}, "none": {}, "variables": {}, "tracked": sorted(ds.__dict__.get("_mtg_tracked", {})),
              "aliases": ds.aliases()}
    tables = {}

    for scale, vids in sorted(by_scale.items()):
        vids = np.asarray(vids, dtype=np.int64)
        name = names[scale]
        relation = scale == connection
        reserved = _RELATION_RESERVED if relation else _GRAPH_RESERVED
        listed = vids.tolist()
        raw_complex = np.array([g._complex.get(v, -1) for v in listed], dtype=np.int64)
        raw_parent = np.array([-1 if g._parent.get(v) is None else g._parent[v] for v in listed], dtype=np.int64)
        columns = {"id": vids}
        present, codes = _lookup(label_property, vids)
        codes = np.array([np.nan if code is None else code for code in codes], dtype=float)
        if present.any():
            names_of = {}
            for code in np.unique(codes[present]):
                label = translator.get(int(code), f"label_{int(code)}")
                known = extras["label_codes"].setdefault(name, {})
                if label in known and known[label] != code:     # the same name for two codes at this scale
                    label = f"{label}_{int(code)}"
                known[label] = int(code) if float(code).is_integer() else float(code)
                names_of[float(code)] = label
            columns["relation" if relation else "label"] = np.array(
                [names_of[float(code)] if here else "_none" for here, code in zip(present, codes)], dtype=object)
        if relation:
            columns["source"] = _lookup(properties.get("n_id_a", {}), vids)[1].astype(np.int64)
            columns["target"] = _lookup(properties.get("n_id_b", {}), vids)[1].astype(np.int64)
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
            columns["edge_type"] = np.array(["none" if not same else ("branch" if code == 3 else "successor")
                                             for same, code in zip(same_scale, edge_codes)], dtype=object)
        columns["metafspm_complex"], columns["metafspm_parent"] = raw_complex, raw_parent
        raw_edges = [edge_property.get(v) for v in listed]
        if not set(raw_edges) <= {None, "/", "<", "+"}:
            raise ValueError(f"{name}: edge types other than '/', '<' and '+' cannot be exchanged")
        columns["metafspm_edge_type"] = np.array([{"/": 1, "<": 2, "+": 3}.get(edge, 0) for edge in raw_edges],
                                                 dtype=np.uint8)

        for property_name, prop in g.properties().items():
            if property_name in ("label", "edge_type") or (relation and property_name in ("n_id_a", "n_id_b")):
                continue
            present, values = _lookup(prop, vids)
            if not present.any():
                continue
            column = f"mtg_{property_name}" if property_name in reserved or property_name in LINK_COLUMNS \
                else property_name
            if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", column):
                warnings.warn(f"MPG property '{property_name}' is not exchanged: its name is not an identifier")
                continue
            if isinstance(prop, ArrayDict):
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
                "container": "ArrayDict" if isinstance(prop, ArrayDict) else "dict", "column": column})
            entry.setdefault("dtypes", {})[name] = kind
            if not present.all():
                extras["absent"][f"{name}.{property_name}"] = vids[~present].tolist()
            none = ([] if isinstance(prop, ArrayDict) else
                    [v for v, here, value in zip(listed, present, values) if here and value is None])
            if none:
                extras["none"][f"{name}.{property_name}"] = none
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
        for name, values in store.items():
            column = f"metafspm_{name}" if name in reserved or name in LINK_COLUMNS else name
            try:
                values, encoding = _object_values(values)
            except TypeError as error:
                warnings.warn(f"variable '{name}' is not exchanged: {error}")
                continue
            if column in columns and not _holds(columns[column], rows, values):
                column = f"metafspm_{name}"         # an MPG property of that name holds other values or text
            _place(columns, column, rows, values, columns["id"].size)
            extras["variables"][name] = dict(_json_meta(ds._variable_meta().get(name, {})), location=location,
                                             **({"column": column} if column != name else {}),
                                             **({"encoding": encoding} if encoding else {}))
    scalars = _scalar_table(ds, extras)
    extras["tracked"] = [name for name in extras["tracked"] if name in extras["variables"]]
    extras.update(_anchor_and_empty_properties(g, anchors))

    entity_sets, relations = [], []
    for scale, columns in sorted(tables.items()):
        if scale == connection:
            endpoints = {g._scale.get(int(v)) for v in np.concatenate([columns["source"], columns["target"]])}
            endpoint = names[endpoints.pop()] if len(endpoints) == 1 else names[g.scales.Compartment]
            relations.append(fspm_tf.Relation(names[scale], columns=columns, source_set=endpoint,
                                              target_set=endpoint))
        else:
            entity_sets.append(fspm_tf.EntitySet(names[scale], columns=columns, rank=int(scale)))
    derived = set(ds.derived())
    for table in entity_sets + relations + ([scalars] if scalars is not None else []):
        _declare(table, metadata, standard_names, derived)
        for column in ("isanchor", "scale"):
            if column in table.columns and column not in table.variables:
                table.variables[column] = fspm_tf.Variable(unit="1", kind="descriptor", role="setting",
                                                           description=f"metafspm: the MPG property {column}")
        for column in LINK_COLUMNS:
            if column in table.columns:
                table.variables[column] = fspm_tf.Variable(
                    unit="1", kind="descriptor", role="setting",
                    description={"metafspm_complex": "metafspm: the MPG's own complex of the vertex (-1: none)",
                                 "metafspm_parent": "metafspm: the MPG's own parent of the vertex, at any scale "
                                                    "(-1: none)",
                                 "metafspm_edge_type": "metafspm: the MPG's edge type (0 none, 1 '/', 2 '<', "
                                                       "3 '+')"}[column])
    meshes = _meshes(ds, entity_sets, scale_of, names, components) if geometry else []
    return fspm_tf.DataStructure(identifier, kind="graph", entity_sets=entity_sets, relations=relations,
                                 scalars=scalars, meshes=meshes, platform=PLATFORM, model="MPGDataStructure",
                                 extras={EXTRAS: extras})


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
    return _fspm_tf().Table("scalars", columns=columns) if columns else None


def _anchor_and_empty_properties(g, anchors: set) -> dict:
    """What the tables cannot hold: the property values of the scale anchors that differ from a new MPG's, the
    properties without any value, and those of a new MPG that this one lacks."""
    fresh = MPG()
    anchor_values, empty = {}, {}
    for name, prop in g.properties().items():
        container = "ArrayDict" if isinstance(prop, ArrayDict) else "dict"
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


def _meshes(ds, entity_sets: list, scale_of: dict, names: dict, components=()) -> list:
    """The DataStructure's triangles (its geometry provider's, those light models get), and each component's
    display_triangles() (shapes drawn for display only, e.g. GRANAP's sections), one mesh each."""
    drawn = []
    try:
        drawn.append(("triangles", ds.triangles()))
    except (ValueError, NotImplementedError):
        pass
    for component in components:
        display = getattr(component, "display_triangles", None)
        if callable(display):
            drawn.append((type(component).__name__, display()))
    return [_mesh_of(ds, entity_sets, scale_of, names, name, triangles) for name, triangles in drawn
            if len(triangles)]


def _mesh_of(ds, entity_sets: list, scale_of: dict, names: dict, name: str, triangles):
    fspm_tf = _fspm_tf()
    sets = {entity_set.name: entity_set for entity_set in entity_sets}
    node_ids = np.asarray(ds.entity_ids("node"), dtype=np.int64)
    owners = np.repeat(triangles.owner, 3)
    feature_rows = {}
    node_set = sets[names[scale_of["node"]]]
    feature_rows[node_set.name] = _rows_of(node_set.columns["id"], node_ids[owners])
    for location in ds._coarse_scale_names():
        set_name = names[scale_of[location]]
        if set_name not in sets or set_name == node_set.name:
            continue
        entity = ds.owner(location)[owners]
        entity_ids = np.asarray(ds.entity_ids(location), dtype=np.int64)
        rows = np.full(owners.size, NO_ID, dtype=np.int64)
        known = entity >= 0
        rows[known] = _rows_of(sets[set_name].columns["id"], entity_ids[entity[known]])
        feature_rows[set_name] = rows
    return fspm_tf.Mesh(name, positions=triangles.vertices.reshape(-1, 3), feature_rows=feature_rows)


# ── Grids → data structures ────────────────────────────────────────────────

def _grid_data_structure(identifier: str, ds, metadata: dict, standard_names: dict):
    fspm_tf = _fspm_tf()
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
    cell_set = fspm_tf.EntitySet("cell", columns=cells)
    face_relation = fspm_tf.Relation("face", columns=faces, source_set="cell", target_set="cell")
    scalar_table = fspm_tf.Table("scalars", columns=scalars) if scalars else None
    derived = set(ds.derived())
    for table in [cell_set, face_relation] + ([scalar_table] if scalar_table is not None else []):
        _declare(table, metadata, standard_names, derived)
    construction = ds._construction()
    grid = fspm_tf.Grid(kind="adaptive" if adaptive else "regular", shape=tuple(construction["shape"]),
                        origin=tuple(construction["origin"]), spacing=tuple(construction["dx"]),
                        axes=tuple(ds.axes), periodic=tuple(bool(p) for p in construction["periodic"]),
                        z_up=False, max_level=int(construction.get("max_level", 0)), cells="cell")
    return fspm_tf.DataStructure(identifier, kind="adaptive_grid" if adaptive else "regular_grid",
                                 entity_sets=[cell_set], relations=[face_relation] if edges.size else [],
                                 scalars=scalar_table, grid=grid, platform=PLATFORM, model=type(ds).__name__,
                                 extras={EXTRAS: extras})


# ── Frames ─────────────────────────────────────────────────────────────────

def to_frame(data_structures: dict, components: dict = None, mappings: dict = None, time: float = 0.,
             step: int = 0, time_unit: str = "s", geometry: bool = True, standard_names: dict = None):
    """
    The fspm-TF frame of *data_structures* ({id: DataStructure}; ids are identifiers), their variables declared
    by *components* ({id: [components]}), with *mappings* ({name: (CrossMapping, source id, target id)}), at *time*
    (in *time_unit*) and *step*. *geometry*: the MPGDataStructures' triangles as meshes. *standard_names*: {metafspm
    name: standard name} on top of the vocabulary's metafspm column (a None value drops one).
    """
    fspm_tf = _fspm_tf()
    from openalea.fspm_tf import vocabulary
    names = dict(vocabulary.platform_names("metafspm"), **(standard_names or {}))
    names = {key: value for key, value in names.items() if value}
    frame = fspm_tf.Frame(time=float(time), time_unit=time_unit, step=int(step),
                          writer={"platform": PLATFORM}, vocabulary_version=vocabulary.version())
    for identifier, ds in data_structures.items():
        metadata = declarations((components or {}).get(identifier, ()))
        if isinstance(ds, MPGDataStructure):
            frame.data_structures.append(_mpg_data_structure(identifier, ds, metadata, names, geometry,
                                                             (components or {}).get(identifier, ())))
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
        frame.mappings.append(fspm_tf.Mapping(
            name, source=(source, source_name), target=(target, "cell"), method=getattr(mapping, "method", None),
            columns={"source": np.asarray(source_ds.entity_ids("node"), dtype=np.int64)[rows],
                     "target": np.asarray(target_ds.entity_ids("cell"), dtype=np.int64)[cells],
                     "weight": np.asarray(weights, dtype=np.float64)}))
    return frame


def scene_to_frame(scene, geometry: bool = True, standard_names: dict = None):
    """The frame of a Scene now: each population's DataStructure (id: the population's name), each environment
    model's (id: the model's class name, numbered when it has several), the scene's CrossMappings."""
    data_structures, components = {}, {}
    for population in scene.populations:
        data_structures[population.name] = population.data_structure
        components[population.name] = population.components
    for model in scene.environment:
        found = [ds for ds in scene._data_structures(model)                 # a population's is written once
                 if not any(ds is other for other in data_structures.values())]
        for number, ds in enumerate(found):
            identifier = type(model).__name__ + (f"_{number}" if len(found) > 1 else "")
            data_structures[identifier] = ds
            components[identifier] = list(model.components)
    identifier_of = {id(ds): identifier for identifier, ds in data_structures.items()}
    mappings = {}
    for mapping in getattr(scene, "mappings", ()):
        if not hasattr(mapping, "incidence"):
            warnings.warn(f"a {type(mapping).__name__} is not exchanged (CrossMappings are)")
            continue
        source, target = identifier_of.get(id(mapping.source)), identifier_of.get(id(mapping.target))
        if source is None or target is None:
            continue
        mappings[f"{source}_to_{target}"] = (mapping, source, target)
    return to_frame(data_structures, components=components, mappings=mappings, time=scene.time,
                    step=scene.iteration, time_unit="s", geometry=geometry, standard_names=standard_names)


def export_frame(path: str, data_structures: dict, **options) -> None:
    """Write to_frame(data_structures, **options) to the .glb file *path*."""
    _fspm_tf().write(path, to_frame(data_structures, **options))


def export_scene(path: str, scene, **options) -> None:
    """Write scene_to_frame(scene, **options) to the .glb file *path*."""
    _fspm_tf().write(path, scene_to_frame(scene, **options))


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
        if extras["class"] == "MPGDataStructure":
            out[data_structure.id] = _mpg_from(data_structure, extras)
        else:
            out[data_structure.id] = _grid_from(data_structure, extras)
    return out


def import_frame(path: str) -> dict:
    """from_frame of the .glb file *path*."""
    return from_frame(_fspm_tf().read(path))


def _mpg_from(data_structure, extras: dict):
    g = MPG()
    if sorted(set(_anchor_vids(g))) != extras["anchors"]:
        raise ValueError(f"'{data_structure.id}': the MPG's scale anchors differ from the written ones")
    scales = extras["scales"]
    tables = {table.name: table for table in data_structure.entity_sets + data_structure.relations}
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
        column = "relation" if isinstance(table, _fspm_tf().Relation) else "label"
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
            absent = np.isin(ids, extras["absent"].get(f"{name}.{property_name}", ()))
            none = np.isin(ids, extras["none"].get(f"{name}.{property_name}", ()))
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
        if isinstance(table, _fspm_tf().Relation):
            _merge_arrays(g, "n_id_a", [(table.columns["id"], table.columns["source"].astype(np.float64))])
            _merge_arrays(g, "n_id_b", [(table.columns["id"], table.columns["target"].astype(np.float64))])
    for property_name, entry in extras.get("anchor_properties", {}).items():
        prop = g._properties.get(property_name)
        if prop is None:
            prop = g._properties[property_name] = ArrayDict() if entry["container"] == "ArrayDict" else {}
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
        ds.track_mtg(VariableSpec(name=name, location=extras["variables"][name]["location"], scale=meta.get("scale"),
                                  mapping=meta.get("mapping"), weight=meta.get("weight"), kind=meta.get("kind"),
                                  variable_type=meta.get("variable_type"), default=meta.get("default", 0.),
                                  on_grow=meta.get("on_grow", "default"), dtype=meta.get("dtype", float),
                                  shape=tuple(meta.get("shape", ()))))
        ds.mark_mtg_synced(name)
    return ds


def _anchor_vids(g) -> list:
    return [g.root] + [int(v) for v in np.flatnonzero(g.topology_arrays()["is_anchor"])]


def _merge_arrays(g, name: str, parts: list) -> None:
    """Add [(vids, values)] to the MPG's ArrayDict property *name* (created when missing), at once."""
    existing = g._properties.get(name)
    if isinstance(existing, ArrayDict):
        keys, values = [existing.order[:existing.size]], [existing.arr[:existing.size]]
    else:                                              # a new MPG's dict (its anchors' values), or none
        keys = [np.fromiter(existing.keys(), dtype=np.int64)] if existing else []
        values = [np.array(list(existing.values()), dtype=np.float64)] if existing else []
    keys += [np.asarray(vids, dtype=np.int64) for vids, _ in parts]
    values += [np.asarray(part, dtype=np.float64) for _, part in parts]
    g._properties[name] = _arraydict(np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64),
                                     np.concatenate(values) if values else np.zeros(0))


def _merge(g, name: str, container: str, values: dict) -> None:
    """Add *values* ({vid: value}) to the MPG's property *name* (created as an ArrayDict or a dict)."""
    existing = g._properties.get(name)
    if container == "ArrayDict":
        keys = list(existing.keys()) if existing is not None else []
        old = [existing[k] for k in keys] if existing is not None else []
        new_keys = [k for k in values if values[k] is not None]
        g._properties[name] = ArrayDict.from_items(np.asarray(keys + new_keys, dtype=np.int64),
                                                   np.asarray(old + [values[k] for k in new_keys], dtype=np.float64))
    else:
        merged = dict(existing or {})
        merged.update(values)
        g._properties[name] = merged


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
