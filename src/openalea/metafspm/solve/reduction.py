"""
Reduction of graph systems by groups: a graph system whose nodes come in many repeated groups (sections, voxels,
motifs) solved on a reduced form. Each group of the reduced location holds only its kept nodes; its interior is
eliminated exactly, once per kind of group, from a full instance evaluated on a library; its interior's values are
recovered on request. Exact when the eliminated interior is linear in the unknowns; a nonlinear term is an
@explicit_term of the graph system, its nodes kept.

Declared on the plant model, before its structure is built (as nodes= and wiring are):

    class Plant(CompositeModel):
        reduction = Reduction(location="SubOrgan", labels=("RootSegment",), ports=conduits, keep=soil_nodes,
                              kind={"barrier": None, "length": Relative(0.1)},
                              inputs={"osmotic_potential": dict(by="tissue", group="tissue_osmotic_potential",
                                                                kept="conduit_osmotic_potential")},
                              systems=("MechaHydraulics._plant_problem",))

  location, labels   the groups (the entities of a coarse location) and the labels of those reduced (all: ())
  ports, keep        rules on a full SectionTemplate (template -> mask of its local vertices): the nodes joined across
                     groups (the structural model's) and the nodes the functional model keeps (its boundary nodes,
                     the nodes of its explicit terms)
  explicit_edges     a rule giving the Connections kept between kept nodes (those of the explicit terms)
  kind               the group variables a kind depends on, {name: None (its value) | Relative(width) (its classes:
                     a group between two the blend of their kinds, interpolate=True; its nearest, False)}; the
                     plant's parameters and the components' settings are always in it
  inputs             the per-group right-hand-side inputs {node variable: {"by": a template property (its categories
                     over the interior), "group": the group's {category: value} input, "kept": the kept nodes' input
                     (NaN: the kind's)}}, acting through the graph system's @input_jacobian(variable)
  systems            the graph systems reduced, "Component._system"
  outputs            the group variables a kind gives its groups (their values on its library's full group, e.g. a
                     section's kr): Reduction.outputs(ds, name)

The kinds of classes are kept once evaluated while their family (template, exact values, parameters) is in use: they
saturate; an exact kind no group has any more is released. The missing kinds are evaluated in batches of at most
Compute.current().kind_batch interior cells, in one library population per plant parameters kept on the Reduction;
the kinds are checkpointed (shared parts, saved once), so a resumed run evaluates none again.

A value a kind derives (a section's condensed volumes, say) is a method of the reduced system's component tagged
@kind_derived(name) (solve.decorator): run once per kind on the library, while the kind's interior factorisation
is live (KronReduction.shares, recover), its value in kind.outputs[name]. A kind's interior is factorised once.

The structural model asks for a group's reduced template (Reduction.template_for) when it instances its anatomy:
the full cells never enter the plant's DataStructure. For display only (an encoding3d frame's sections, drawn whole),
Reduction.recover_interiors gives every reduced group's interior values at once (one factorisation per kind),
full_rows the full template's rows a reduced one holds, and display_ids the ids the interior's vertices are written
under (never a vertex id). A kind: (the full template, the kind variables' values or
classes, the plant's parameters and the settings), evaluated on a library population of the plant model (its own
components, those of the reduced systems and of their inputs at the cells run): the graph system's own residual and
Jacobian (decorator._EXTRACTING: without its boundary sets and its explicit terms), on a full instance and on a
reduced one; the group's term is the full one's Kron reduction minus
what the model computes on the reduced structure itself, so that every term is counted once, whether the model
computes it live or the kind holds it.
"""
import copy
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from openalea.metafspm.solve.network import kron_reduction

# The ids of a reduced group's interior vertices in an output (an encoding3d frame's display rows: they are not the
# MPG's vertices): DISPLAY_ID_BASE + the group's vertex × DISPLAY_ROWS + the vertex's row in the full template, the
# same from one frame to the next, above every vertex id
DISPLAY_ID_BASE, DISPLAY_ROWS = 2 ** 52, 2 ** 24


def display_ids(owner: int, rows) -> np.ndarray:
    """The display ids of the full template's *rows* of the reduced group *owner* (its vertex): its interior's
    vertices, as an output writes them (never an MPG vertex id)."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size and (rows.min() < 0 or rows.max() >= DISPLAY_ROWS):
        raise ValueError(f"display_ids: a full template's rows are below {DISPLAY_ROWS}")
    return DISPLAY_ID_BASE + int(owner) * DISPLAY_ROWS + rows


def is_display_id(ids) -> np.ndarray:
    """Whether each of *ids* is a display id (display_ids), not an MPG vertex id."""
    return np.asarray(ids, dtype=np.int64) >= DISPLAY_ID_BASE


@dataclass(frozen=True)
class Relative:
    """
    Classes of relative width *width*: the values (1 + width)^k. With *interpolate* (default), a value between two
    classes is given both, weighted linearly by its place between them (its group's terms the two kinds' blend: exact
    at the class values and for terms affine in the value, a curvature's error width²/8 of it between them); without,
    its nearest class (its kind's terms as they are: an error up to width times the terms' change).
    """
    width: float
    interpolate: bool = True

    def __post_init__(self):
        if not float(self.width) > 0:
            raise ValueError(f"Relative: width is positive, got {self.width!r}")

    def classify(self, value) -> float:
        """The nearest class of *value* (a value not positive or not finite: itself)."""
        value = float(value)
        if value <= 0 or not np.isfinite(value):
            return value
        base = np.log1p(float(self.width))
        return float(np.exp(np.round(np.log(value) / base) * base))

    def corners(self, value) -> tuple:
        """((weight, class), ...) of *value*: its two neighbouring classes weighted by its place between them (one at a
        class value), or its nearest class without interpolation (a value not positive or not finite: itself)."""
        value = float(value)
        if not self.interpolate or value <= 0 or not np.isfinite(value):
            return ((1., self.classify(value)),)
        base = np.log1p(float(self.width))
        k = np.floor(np.log(value) / base)
        low, high = float(np.exp(k * base)), float(np.exp((k + 1) * base))
        weight = (value - low) / (high - low)
        if weight <= CLASS_ROUNDING:
            return ((1., low),)
        if weight >= 1. - CLASS_ROUNDING:
            return ((1., high),)
        return ((1. - weight, low), (weight, high))


CLASS_ROUNDING = 1e-12          # a value within this fraction of the class interval from a class: that class alone


@contextmanager
def _extracting():
    from openalea.metafspm.solve import decorator
    decorator._EXTRACTING.append(True)
    try:
        yield
    finally:
        decorator._EXTRACTING.pop()


@dataclass
class Kind:
    """
    A kind's reduction: its term on the kept nodes (D, e), the inputs' responses, and its interior for recovery. A
    blend (a group between two classes of a Relative variable: Reduction.group_kinds) has its parts ((weight, Kind),
    ...): its D, e and outputs their weighted sums, its inputs' terms and its interior its parts' weighted.
    Checkpointed as a shared part (scene.shared), saved once: a resumed run evaluates no kind again.
    """
    D: np.ndarray
    e: np.ndarray
    reduction: object
    interior_source: np.ndarray
    kept_rows: np.ndarray
    interior_rows: np.ndarray
    inputs: dict
    outputs: dict = None
    key: tuple = None                            # its key in its Reduction's kinds: (reduced template key, kind
                                                 # values, plant parameters' signature), the same when evaluated again
    system: str = None                           # the system it reduces
    parts: tuple = None                          # a blend's ((weight, Kind), ...)

    shared_in_checkpoints = "kind"               # class attribute (not a field): scene.shared's part name

    def release(self) -> None:
        """Free its arrays: a kind no group of its DataStructure has any more (Reduction.group_kinds' release), its
        memory freed even where a cache still holds it. Its derived values (@kind_derived, in its outputs) go with
        it, computed again with it."""
        self.D = self.e = self.reduction = self.interior_source = None
        self.kept_rows = self.interior_rows = None
        self.inputs, self.outputs, self.parts = {}, {}, None

    def recover(self, kept_values) -> np.ndarray:
        """Its interior's values (interior_rows' order) for the kept nodes' *kept_values* (a vector, or a matrix of
        columns): its interior's solve (one factorisation), a blend's its parts' weighted."""
        x = np.asarray(kept_values, dtype=float)
        if self.parts:
            return sum(weight * part.recover(x) for weight, part in self.parts)
        source = -np.asarray(self.interior_source, dtype=float)
        return np.asarray(self.reduction.recover(x, source[:, None] if x.ndim == 2 else source))


def _blend(parts) -> Kind:
    """The Kind of a group between classes: *parts* ((weight, Kind), ...: the weights summing to 1) blended, its D, e
    and numeric outputs weighted (an output that is not a number: the heaviest part's)."""
    weights = [weight for weight, _ in parts]
    heaviest = parts[int(np.argmax(weights))][1]
    first = parts[0][1]
    outputs = {}
    for name, value in (heaviest.outputs or {}).items():
        try:
            values = [np.asarray(part.outputs[name], dtype=float) for _, part in parts]
            blended = sum(weight * v for weight, v in zip(weights, values))
            outputs[name] = float(blended) if np.ndim(blended) == 0 and not isinstance(value, np.ndarray) else blended
        except (KeyError, TypeError, ValueError):
            outputs[name] = value
    return Kind(D=sum(weight * part.D for weight, part in parts), e=sum(weight * part.e for weight, part in parts),
                reduction=None, interior_source=None, kept_rows=first.kept_rows, interior_rows=first.interior_rows,
                inputs={}, outputs=outputs, key=None, system=first.system, parts=tuple(parts))


class Reduction:
    def __init__(self, location: str = "SubOrgan", labels=(), ports=None, keep=None, explicit_edges=None,
                 kind: dict = None, inputs: dict = None, systems=(), outputs=()):
        self.location, self.labels = location, tuple(labels)
        self.ports, self.keep, self.explicit_edges = ports, keep, explicit_edges
        self.kind, self.inputs, self.systems = dict(kind or {}), dict(inputs or {}), tuple(systems)
        self.output_names = tuple(outputs)
        self._reduced, self._full, self._kept, self._rows = {}, {}, {}, {}
        self._kinds, self._blends, self._libraries = {}, {}, {}
        self.model, self.instance, self.g = None, None, None
        self.library = False                     # a library's: its groups' kinds are not evaluated (no library of
                                                 # its own)

    def __getstate__(self):
        state = dict(self.__dict__)        # a checkpoint: the declaration, the templates and the kinds (shared parts)
        state.update(_blends={}, _libraries={}, model=None, instance=None, g=None)
        return state

    def __setstate__(self, state):
        state.setdefault("_blends", {})
        state.setdefault("_libraries", {})
        self.__dict__.update(state)

    # ── The population's own (Scene, population_data_structure) ──

    def for_population(self, g) -> "Reduction":
        """A copy for one population's MPG *g* (the templates shared, the kinds its own)."""
        out = copy.copy(self)
        out._kinds, out._blends, out._libraries, out.g = {}, {}, {}, g
        return out

    def bind(self, model, instance) -> None:
        """The plant model (its class, for the library) and its instance (its components' parameters)."""
        self.model, self.instance = model, instance

    # ── Structure ──

    def reduces(self, label_name) -> bool:
        """Whether the groups labelled *label_name* are reduced."""
        return not self.labels or label_name in self.labels

    def template_for(self, full):
        """The reduced template of the full SectionTemplate *full*: its kept nodes (ports and keep), their Cells and
        Layers, and the explicit edges between them; one per full template."""
        if full.key in self._reduced:
            return self._reduced[full.key]
        kept, held = self._restriction(full)
        reduced = full.restricted(held, key=tuple(np.atleast_1d(full.key)) + ("reduced",))
        self._reduced[full.key], self._full[reduced.key] = reduced, full
        self._kept[reduced.key] = np.flatnonzero(kept)
        self.__dict__.setdefault("_rows", {})[reduced.key] = np.flatnonzero(held)
        return reduced

    def _restriction(self, full) -> tuple:
        """(the kept Compartments, every vertex the reduced template holds), masks over the full template *full*: the
        kept nodes (ports and keep), their Cells and Layers, the explicit edges."""
        from openalea.metafspm.data_structure.configs import ScalesConfig
        compartment = full.scale == ScalesConfig.Compartment
        kept = np.zeros(full.size, dtype=bool)
        for rule in (self.ports, self.keep):
            if rule is not None:
                kept |= np.asarray(rule(full), dtype=bool)
        kept &= compartment
        structure = np.zeros(full.size, dtype=bool)
        above = full.complex[kept & (full.complex >= 0)]
        while above.size:
            structure[above] = True
            above = full.complex[above]
            above = above[above >= 0]
        edges = np.zeros(full.size, dtype=bool)
        if self.explicit_edges is not None:
            edges = np.asarray(self.explicit_edges(full), dtype=bool) & (full.scale == ScalesConfig.Connection)
        return kept, kept | structure | edges

    def full_rows(self, key) -> np.ndarray:
        """The rows in its full template of each vertex of the reduced template of key *key* (template_for's, in the
        reduced template's order)."""
        rows = self.__dict__.setdefault("_rows", {})        # a checkpoint of an earlier version: recomputed
        if key not in rows:
            rows[key] = np.flatnonzero(self._restriction(self._full[key])[1])
        return rows[key]

    def full_template(self, key):
        """The full SectionTemplate whose reduced template has the key *key* (a KeyError for a key template_for did
        not give)."""
        return self._full[key]

    def reduced_groups(self, ds) -> dict:
        """{group entity index: (its vertex, its reduced template key)} of the DataStructure's reduced groups."""
        g = ds.structure
        if not g._instanced():
            return {}
        instances = g.instances
        ids = ds.entity_ids(self.location)
        index = {int(v): i for i, v in enumerate(ids.tolist())}
        out = {}
        for owner, key in zip(instances.owner.tolist(), instances.template):
            if key in self._full and owner in index:
                out[index[owner]] = (owner, key)
        return out

    # ── The terms of a graph system's solve ──

    def terms(self, component, system: str, ds, node_ids) -> tuple:
        """(D, e) on the solved nodes *node_ids*: the reduced groups' interiors' terms (sparse, nodes × nodes, and a
        vector), or None when no solved group is reduced."""
        groups = self.reduced_groups(ds)
        if not groups or self.library:
            return None
        n = node_ids.size
        position = {int(v): i for i, v in enumerate(node_ids.tolist())}
        kinds = self.group_kinds(ds, groups, release=True)
        rows, cols, values, e = [], [], [], np.zeros(n)
        g = ds.structure
        for i, (owner, key) in groups.items():
            first, _ = g.instances.block(owner)
            kept = first + np.flatnonzero(self._reduced_scale(key) == g.scales.Compartment)
            local = np.array([position.get(int(v), -1) for v in kept.tolist()])
            if (local < 0).any():
                continue                                   # the group is not solved (a filter)
            kind = kinds[i][system]
            e[local] += kind.e + self._input_terms(ds, kind, i, kept, owner)
            r, c = np.meshgrid(local, local, indexing="ij")
            rows.append(r.ravel()), cols.append(c.ravel()), values.append(kind.D.ravel())
        if not rows:
            return None
        D = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        return D, e

    def term_values(self, component, system: str, ds, values) -> np.ndarray:
        """D x + e of *system*'s reduced groups for the node values *values* (DataStructure order): the interiors'
        part of each kept node's balance (e.g. a soil node's uptake), 0 elsewhere."""
        out = np.zeros(ds.n_nodes())
        if self.library:
            return out
        terms = self.terms(component, system, ds, ds.entity_ids("node"))
        if terms is not None:
            D, e = terms
            out = np.asarray(D @ np.asarray(values, dtype=float)).reshape(-1) + e
        return out

    def outputs(self, ds, name: str) -> np.ndarray:
        """Group variable *name* of each group of the location: its kind's (its library full group's value) for a
        reduced group, NaN elsewhere."""
        out = np.full(len(ds.entity_ids(self.location)), np.nan)
        groups = self.reduced_groups(ds)
        if not groups or self.library:
            return out
        kinds = self.group_kinds(ds, groups)
        for i, by_system in kinds.items():
            kind = next(iter(by_system.values()))
            out[i] = (kind.outputs or {}).get(name, np.nan)
        return out

    def _reduced_scale(self, key):
        return self._reduced[self._full[key].key].scale

    def _input_terms(self, ds, kind, group, kept, owner) -> np.ndarray:
        """The change of a group's kept sources for its own inputs: the kept nodes' (kept input, NaN: the kind's)
        and its categories' ({category: value} group input)."""
        if kind.parts:
            return sum(weight * self._input_terms(ds, part, group, kept, owner) for weight, part in kind.parts)
        out = np.zeros(kind.D.shape[0])
        for name, response in kind.inputs.items():
            declared = self.inputs[name]
            if declared.get("kept") and ds.has(declared["kept"]):
                values = np.asarray(ds.get(declared["kept"]), dtype=float)[ds.index_of(kept)]
                delta = np.where(np.isfinite(values), values - response["base_kept"], 0.)
                out += response["kept"] @ delta
            if declared.get("group") and ds.has(declared["group"]):
                given = np.asarray(ds.get(declared["group"]), dtype=object).reshape(-1)
                chosen = given[group] if group < given.size else None
                for category, value in (chosen or {}).items():
                    if category in response["categories"]:
                        unit, base = response["categories"][category]
                        out += float(value) * unit - base
        return out

    # ── Kinds ──

    def _kind_corners(self, ds, groups) -> dict:
        """{group: ((weight, the kind variables' values or classes), ...)}: one kind's values, or the corners of its
        classes it lies between (Relative(interpolate=True): two per such variable, their weights' products)."""
        per_variable = {i: [] for i in groups}
        owner = None
        for name, classes in self.kind.items():
            values = np.asarray(ds.get(name))
            if ds.location(name) == "node":
                if owner is None:
                    owner = np.asarray(ds.owner(self.location))
                first = _first_nodes(owner)
                picked = {i: float(values[first[i]]) for i in groups}
            else:
                picked = {i: float(values[i]) for i in groups}
            for i, value in picked.items():
                per_variable[i].append(((1., value),) if classes is None else classes.corners(value))
        out = {}
        for i, choices in per_variable.items():
            corners = [(1., ())]
            for options in choices:
                corners = [(weight * w, values + (v,)) for weight, values in corners for w, v in options]
            out[i] = tuple(corners)
        return out

    def _parameters(self, ds, plant_index) -> dict:
        """The plant's parameters and the components' settings ({name: value}), for its kinds' key and library."""
        from dataclasses import fields
        out = {}
        for component in self.instance.components:
            for f in fields(component):
                meta = f.metadata
                if meta.get("variable_type") == "setting":
                    out[f.name] = getattr(component, f.name)
            located = {f.name for f in fields(component)
                       if f.metadata.get("scale") is not None or f.metadata.get("location") is not None}
            for name in type(component)._plant_parameter_fields():
                if name in located:              # per entity (an edge's, a node's): the DataStructure's, not a kind's
                    continue
                try:
                    values = np.asarray(component.parameter_values(name, "Plant"), dtype=float).reshape(-1)
                except (KeyError, ValueError, AttributeError):
                    values = np.asarray([getattr(component, name)], dtype=float)
                out[name] = float(values[plant_index] if values.size > plant_index else values[0])
            for f in fields(component):                  # its Plant-scale parameters, any value (a dict, a boolean)
                if (f.metadata.get("variable_type") == "parameter" and f.name not in out
                        and f.metadata.get("scale") == _plant_scale() and ds.has(f.name)
                        and ds.location(f.name) == "Plant"):
                    values = ds.get(f.name)
                    value = values[plant_index] if len(values) > plant_index else values[0]
                    out[f.name] = value.item() if isinstance(value, np.generic) else value
        return out

    def group_kinds(self, ds, groups, release: bool = False) -> dict:
        """
        {group: {system: Kind}}: the kinds of *groups*, the missing ones evaluated (per plant's parameters, in
        batches: _evaluate_missing); a group between two classes of a Relative variable gets their blend (_blend,
        kept while its weights are the same). *release*: *groups* are all the DataStructure's reduced groups (a
        solve's terms), and the kinds none of them has are released (their memory freed, evaluated again if a group
        comes back to one), except a class's whose exact kind values, template and parameters a current group has (a
        growing section's former classes: kept, a bounded set, so that the kinds saturate). Without classes the kinds
        kept are those of the plant's current groups only.
        """
        corners = self._kind_corners(ds, groups)
        plant = np.asarray(ds.owner("Plant"))
        owner = np.asarray(ds.owner(self.location))
        first = _first_nodes(owner)
        by_plant, keys, missing = {}, {}, {}
        for i, (vertex, key) in groups.items():
            p = int(plant[first[i]])
            if p not in by_plant:
                parameters = self._parameters(ds, p)
                by_plant[p] = (parameters, _signature(parameters))
            parameters, signature = by_plant[p]
            keys[i] = tuple((weight, (key, values, signature)) for weight, values in corners[i])
            for _, kind_key in keys[i]:
                if kind_key not in self._kinds:
                    missing.setdefault(signature, (parameters, {}))[1].setdefault(kind_key, (vertex, key,
                                                                                             kind_key[1]))
        for parameters, requests in missing.values():
            self._evaluate_missing(parameters, requests)
        blends = self.__dict__.setdefault("_blends", {})
        out = {}
        for i, chosen in keys.items():
            if len(chosen) == 1:
                out[i] = self._kinds[chosen[0][1]]
                continue
            if chosen not in blends:
                blends[chosen] = {system: _blend([(weight, self._kinds[kind_key][system])
                                                  for weight, kind_key in chosen])
                                  for system in self._kinds[chosen[0][1]]}
            out[i] = blends[chosen]
        if release:
            used = {kind_key for chosen in keys.values() for _, kind_key in chosen}
            exact = [position for position, classes in enumerate(self.kind.values()) if classes is None]
            current = {self._class_family(kind_key, exact) for kind_key in used}
            for kind_key in [k for k in self._kinds if k not in used]:
                if len(exact) < len(self.kind) and self._class_family(kind_key, exact) in current:
                    continue                              # a class of a current group's family: kept
                for kind in self._kinds.pop(kind_key).values():
                    kind.release()
            for chosen in [c for c in blends if c not in set(keys.values())]:
                del blends[chosen]
        return out

    @staticmethod
    def _class_family(kind_key, exact) -> tuple:
        """A kind key without its classed values: (template key, the exact kind values, signature)."""
        key, values, signature = kind_key
        return key, tuple(values[position] for position in exact), signature

    def _evaluate_missing(self, parameters: dict, requests: dict) -> None:
        """Evaluate the kinds *requests* ({kind key: (a representative group vertex, its reduced template key, its kind
        values)}) of a plant of *parameters*, in batches of at most Compute.current().kind_batch interior cells (one
        kind at least), the library's transients freed between batches (Compute.trim)."""
        from openalea.metafspm.compute import Compute
        compute = Compute.current()
        batch, cells = {}, 0
        items = list(requests.items())
        for position, (kind_key, request) in enumerate(items):
            size = self._interior_cells(request[1])
            if batch and cells + size > compute.kind_batch:
                self._kinds.update(_evaluate(self, parameters, batch))
                compute.trim()
                batch, cells = {}, 0
            batch[kind_key], cells = request, cells + size
        if batch:
            self._kinds.update(_evaluate(self, parameters, batch))

    def _interior_cells(self, key) -> int:
        """The interior cells of the reduced template of key *key* (its full template's Compartments not kept)."""
        from openalea.metafspm.data_structure.configs import ScalesConfig
        full = self._full[key]
        return int(np.count_nonzero(full.scale == ScalesConfig.Compartment)) - int(self._kept[key].size)

    def unknown(self, system: str = None) -> str:
        """The node unknown of the reduced *system* (default: the first): its graph system's (one node unknown)."""
        from openalea.metafspm.solve.decorator import declared_system
        system = system or self.systems[0]
        attr = system.split(".")[1]
        component = next(c for c in self.instance.components if declared_system(c, attr, (system,)) is not None)
        return type(component)._graph_system_specs[attr]["node_unknowns"][0]

    def recover_interiors(self, ds, groups=None, system: str = None) -> dict:
        """
        {group: (its vertex, its reduced template key, its full template, its interior's Compartment rows in it, the
        unknown of *system* there)} of the reduced *groups* (entity indices of the location; default: all), from their
        kept nodes' values in *ds* (recover's, the same values): batched by kind, one factorisation of each kind's
        interior for all its groups (its columns at once), freed after. For outputs (a frame's sections drawn whole):
        the DataStructure stays reduced.
        """
        every = self.reduced_groups(ds)
        if not every or self.library:
            return {}
        chosen = every if groups is None else {int(i): every[int(i)] for i in groups if int(i) in every}
        system = system or self.systems[0]
        values = np.asarray(ds.get(self.unknown(system)), dtype=float)
        kinds = self.group_kinds(ds, chosen)
        g = ds.structure
        by_kind = {}
        for i, (owner, key) in chosen.items():
            kind = kinds[i][system]
            first, _ = g.instances.block(owner)
            kept = first + np.flatnonzero(self._reduced_scale(key) == g.scales.Compartment)
            column = values[ds.index_of(kept)]
            for weight, part in (kind.parts or ((1., kind),)):              # a blend's parts, each solved once
                by_kind.setdefault(id(part), (part, []))[1].append((i, weight, column))
        out = {}
        for part, members in by_kind.values():
            interior = part.recover(np.column_stack([member[2] for member in members]))
            interior = np.asarray(interior).reshape(len(part.interior_rows), len(members))
            for column, (i, weight, _) in enumerate(members):
                if i in out:
                    out[i][4][...] += weight * interior[:, column]
                else:
                    owner, key = chosen[i]
                    out[i] = (owner, key, self._full[key], part.interior_rows, weight * interior[:, column])
            del interior
        return out

    def recover(self, ds, group: int, values, system: str = None) -> tuple:
        """A reduced group's interior: (its full template's local rows, the values of the system's unknown there),
        from its kind and its kept nodes' *values* (the unknown on them, DataStructure order)."""
        groups = self.reduced_groups(ds)
        kind = self.group_kinds(ds, {group: groups[group]})[group][system or self.systems[0]]
        return kind.interior_rows, kind.recover(values)


def _first_nodes(owner) -> np.ndarray:
    """Each group's first node (its index among the nodes; -1 for a group without any), from the nodes' *owner*
    groups (entity indices)."""
    owner = np.asarray(owner, dtype=np.int64)
    out = np.full(int(owner.max()) + 1 if owner.size else 0, -1, dtype=np.int64)
    owned = owner >= 0
    groups, starts = np.unique(owner[owned], return_index=True)
    out[groups] = np.flatnonzero(owned)[starts]
    return out


def _signature(parameters: dict) -> str:
    """A digest of a plant's parameters, by content (dicts, records and numbers alike)."""
    from openalea.metafspm.solve.memo import content_key
    return content_key({name: parameters[name] for name in sorted(parameters)})


def _plant_scale() -> int:
    from openalea.metafspm.data_structure.configs import ScalesConfig
    return ScalesConfig.Plant


def _library_components(instance, systems) -> list:
    """
    The components a library runs, in the plant model's order: those of the reduced *systems*, and the functional
    components giving them an input at the cells (Compartment or Connection scale: per-cell osmotic potentials from
    their templates, say). The others are not: their inputs at coarser scales are the kinds' values or the
    boundaries' (not the library's), their own tables and forcings the plant's (a shoot's development).
    """
    from dataclasses import fields
    from openalea.metafspm.coupling.component import StructuralComponent
    from openalea.metafspm.data_structure.configs import ScalesConfig
    from openalea.metafspm.solve.decorator import declared_system
    functional = [component for component in instance.components if not isinstance(component, StructuralComponent)]
    reducing = [component for component in functional
                if any(declared_system(component, system.split(".")[1], (system,)) is not None for system in systems)]

    def at_cells(f) -> bool:
        scale = f.metadata.get("scale")
        return scale is not None and scale >= ScalesConfig.Compartment

    needed = {f.name for component in reducing for f in fields(component)
              if f.metadata.get("variable_type") == "input" and at_cells(f)}
    return [component for component in functional
            if component in reducing or any(f.metadata.get("variable_type") == "state_variable" and at_cells(f)
                                            and f.name in needed for f in fields(component))]


_STRUCTURAL = ("scale", "edge_type", "label", "isanchor")     # the vertex properties a library group does not copy


class _Library:
    """
    A reduction's library population for one plant's parameters (their signature) and the scene's time step: a
    population of the plant model (its own components, those of the reduced systems and of their inputs at the
    cells run), built once and kept on the Reduction (_libraries, not checkpointed), its components reset to their
    built state at each batch (no state left from the former groups). Each batch of kinds gets one full and one reduced group per kind (a
    representative group's complexes, label and properties, its template's instance); the next batch's replace them
    (removed in the same update of the library's DataStructure: a library of no group would have no plant).
    """

    def __init__(self, reduction: Reduction, parameters: dict, time_step: float):
        self.reduction, self.parameters, self.time_step = reduction, parameters, time_step
        self.population, self.model, self.built = None, None, {}
        self.placed = []                 # the last batch's vertices (each group's complexes, then the group)

    def place(self, g, plant, items) -> dict:
        """{group vertex: (k, "full" | "reduced")}: the groups of the kinds *items* placed in the library's MPG *g*
        under *plant*."""
        from openalea.metafspm.data_structure.configs import PropsConfig
        reduction = self.reduction
        plant_g = reduction.g
        scale = getattr(plant_g.scales, reduction.location)
        made = {}
        for k, (_, (vertex, key, _)) in enumerate(items):
            chain = []                                     # the representative's complexes, Plant excluded
            v = plant_g.complex(vertex)
            while v is not None and plant_g.scale(v) > plant_g.scales.Plant:
                chain.append(v)
                v = plant_g.complex(v)
            for which, template in (("full", reduction._full[key]),
                                    ("reduced", reduction._reduced[reduction._full[key].key])):
                parent, vertices = plant, []
                for source in reversed(chain):
                    parent = g.add_component(parent, **PropsConfig(scale=plant_g.scale(source), edge_type='/',
                                                                   label=plant_g.label(source)))
                    vertices.append(parent)
                group = g.add_component(parent, **PropsConfig(scale=scale, edge_type='/',
                                                              label=plant_g.label(vertex)))
                for name, prop in plant_g.properties().items():
                    value = prop.get(vertex) if hasattr(prop, "get") else None
                    if value is not None and name not in _STRUCTURAL:
                        g.property(name)[group] = value
                g.add_instance(group, template)
                made[group] = (k, which)
                self.placed.append(vertices + [group])
        return made

    def remove(self, g) -> None:
        """The last batch's groups and their complexes removed from the library's MPG *g* (the DataStructure follows
        at its next update)."""
        for vertices in self.placed:
            for vertex in reversed(vertices):
                g.remove_vertex(vertex)
        self.placed = []

    def groups(self, items) -> dict:
        """{group vertex: (k, which)} of the batch *items*: the population built at the first batch, its groups
        placed in it after."""
        reduction = self.reduction
        if self.population is not None:
            ds = self.population.data_structure
            g = ds.structure
            plant = int(ds.entity_ids("Plant")[0])
            self.remove(g)
            made = self.place(g, plant, items)
            self._reset_components()
            ds.update_topology()
            self._read_structure()
            self._apply_parameters()
            return made
        from openalea.metafspm.coupling.component import StructuralComponent
        from openalea.metafspm.scene.population import Planting, planting, population_data_structure
        library, made = self, {}

        class LibraryStructure(StructuralComponent):
            @classmethod
            def initiate_plant(cls, g, plant, scenario):
                made.update(library.place(g, plant, items))

        self.model = model = type(f"Library{reduction.model.__name__}", (reduction.model,),
                                  dict(initiators=(LibraryStructure,), wiring=(), reduction=None))
        population = population_data_structure(planting([Planting()], model, {}), model,
                                               time_step=self.time_step)       # the scene's step (_library)
        ds, instance = population.data_structure, population.instance
        own = reduction.for_population(ds.structure)         # the library knows its reduced groups, evaluates none
        own.library = True
        own.bind(model, instance)
        ds.structure.reduction = own
        self.population = population
        self._apply_parameters()
        self.built = {id(component): dict(component.__dict__) for component in instance.components}
        return made

    def _reset_components(self) -> None:
        """Each component's state as the library built it (its fields, its construction's records): what a run left
        on it (a solve's previous values, its caches and memos, sized to the former groups) dropped."""
        for component in self.population.instance.components:
            built = self.built[id(component)]
            for name in [name for name in component.__dict__ if name not in built]:
                del component.__dict__[name]
            for name, value in built.items():
                component.__dict__[name] = copy.copy(value) if isinstance(value, (dict, list, set)) else value

    def _read_structure(self) -> None:
        """The variables its MPG holds (its groups' templates' and properties' values) read again, as a library built
        anew reads them (the components' MTG-backed variables): the groups placed since have their own."""
        ds = self.population.data_structure
        aliases, derived = ds.aliases(), ds.derived()
        for component in self.population.instance.components:
            for spec in getattr(component, "_variable_specs", {}).values():
                if (not spec.mtg_backed or not ds.has(spec.name) or spec.name in aliases or spec.name in derived
                        or ds.location(spec.name) != spec.location):    # held at another location by another
                    continue                                            # component (its source, not this one)
                try:
                    values = ds._read_mtg(spec)
                except (TypeError, ValueError):         # a property of another type of the same name: not its
                    continue                            # source (the variable registered by another component)
                if values is not None and np.shape(values) == np.shape(ds.get(spec.name)):
                    ds.set(spec.name, values)

    def _apply_parameters(self) -> None:
        """The plant's parameters and the components' settings given to the library's plant and components."""
        from dataclasses import fields
        ds, instance = self.population.data_structure, self.population.instance
        for component in instance.components:
            names = {f.name for f in fields(component) if f.metadata.get("variable_type") in ("setting", "parameter")}
            for name, value in self.parameters.items():
                if name not in names:
                    continue
                if ds.has(name) and ds.location(name) == "Plant":   # the library's one plant: its value there
                    stored = np.asarray(ds.get(name))
                    values = np.empty(stored.shape, dtype=object) if stored.dtype == object else stored.copy()
                    for i in range(len(values)):
                        values[i] = value                         # a record as it is (a dict, nested lists)
                    ds.set(name, values)
                else:
                    setattr(component, name, value)


def _library(reduction: Reduction, parameters: dict) -> _Library:
    """The reduction's library for a plant of *parameters* at the scene's time step (built at its first batch)."""
    from openalea.metafspm.coupling.choregrapher import Choregrapher
    time_step = Choregrapher().simulation_step()
    time_step = 3600. if time_step is None else float(time_step)
    libraries = reduction.__dict__.setdefault("_libraries", {})
    key = (_signature(parameters), time_step)
    if key not in libraries:
        libraries[key] = _Library(reduction, parameters, time_step)
    return libraries[key]


def _evaluate(reduction: Reduction, parameters: dict, requests: dict) -> dict:
    """{kind key: {system: Kind}} of *requests* ({kind key: (a representative group vertex, its reduced template
    key, its kind values)}) for a plant of *parameters*: one full and one reduced group per kind in the reduction's
    library for those parameters (_Library), isolated, replaced by the next batch's."""
    return _evaluate_in(reduction, _library(reduction, parameters), list(requests.items()))


def _evaluate_in(reduction: Reduction, library: _Library, items: list) -> dict:
    """_evaluate's kinds *items* in *library*."""
    from openalea.metafspm.solve.decorator import _GraphSystemBuilder, declared_system, kind_derived_methods
    plant_g = reduction.g
    groups_made = library.groups(items)
    ds = library.population.data_structure
    instance = library.population.instance
    ids = ds.entity_ids(reduction.location).tolist()
    entity = {int(v): i for i, v in enumerate(ids)}
    group_of = {(k, which): entity[int(vertex)] for vertex, (k, which) in groups_made.items()}
    owner = np.asarray(ds.owner(reduction.location))
    for position, (name, classes) in enumerate(reduction.kind.items()):        # the kinds' values (their classes)
        values = np.array(ds.get(name), dtype=float, copy=True)
        location = ds.location(name)
        for (k, which), i in group_of.items():
            value = items[k][1][2][position]
            if location == "node":
                values[owner == i] = value
            else:
                values[i] = value
        ds.set(name, values)
    from openalea.metafspm.solve import decorator
    reducing = _library_components(instance, reduction.systems)
    decorator._SKIPPED.update(reduction.systems)                 # the reduced systems: read, not solved
    try:
        for component in reducing:
            component()
    finally:
        decorator._SKIPPED.difference_update(reduction.systems)
    out = {kind_key: {} for kind_key, _ in items}
    for system in reduction.systems:
        component_name, attr = system.split(".")
        component = next(c for c in instance.components if declared_system(c, attr, (system,)) is not None)
        spec_def = type(component)._graph_system_specs[attr]
        with _extracting():
            builder = _GraphSystemBuilder(component, spec_def)
            spec, _, _ = builder.build()
            x = spec.pack_unknowns()
            residual = np.asarray(spec.residual(x), dtype=float)
            A = csr_matrix(spec.jacobian(x))
            c = residual - A @ x
            context = spec.make_context(x, None, None)
            input_jacobians = {name: csr_matrix(evaluator(context))
                               for name, evaluator in builder.input_jacobian_evaluators.items()}
        derived = kind_derived_methods(component)
        node_of = {int(v): j for j, v in enumerate(spec.graph.node_ids.tolist())}
        for k, (kind_key, (_, key, _)) in enumerate(items):
            full_entities = np.flatnonzero(owner == group_of[(k, "full")])        # DataStructure indices
            reduced_nodes = np.flatnonzero(owner == group_of[(k, "reduced")])
            full_nodes = np.array([node_of[int(v)] for v in ds.entity_ids("node")[full_entities]])
            reduced_nodes = np.array([node_of[int(v)] for v in ds.entity_ids("node")[reduced_nodes]])
            full_template = reduction._full[key]
            compartments = np.flatnonzero(full_template.scale == plant_g.scales.Compartment)
            kept = np.isin(compartments, reduction._kept[key])
            A_full = A[full_nodes][:, full_nodes]
            kron = kron_reduction(A_full, kept)
            I, K = kron.interior_nodes, kron.kept_nodes
            c_full = c[full_nodes]
            S, s = kron.reduced, c_full[K] + kron.inject(c_full[I])
            A_red, c_red = A[reduced_nodes][:, reduced_nodes].toarray(), c[reduced_nodes]
            inputs = {}
            declared = [name for name in input_jacobians if name in reduction.inputs]
            if declared:                       # −Xᵀ for a symmetric operator (no solve), only for a declared input
                carry = kron.injection()
            for name in declared:
                Jp = input_jacobians[name]
                J_full = Jp[full_nodes][:, full_nodes]
                response = np.asarray(J_full[K].toarray()) + np.asarray((J_full[I].T @ carry.T).T)
                own = Jp[reduced_nodes][:, reduced_nodes].toarray()
                base = np.asarray(ds.get(name), dtype=float)
                base_full = base[np.flatnonzero(owner == group_of[(k, "full")])]
                categories = {}
                by = reduction.inputs[name].get("by")
                if by is not None and by in full_template.properties:
                    rows, values = full_template.properties[by]
                    label = np.full(full_template.size, None, dtype=object)
                    label[rows] = values
                    hidden = label[compartments][I]
                    for category in sorted({t for t in hidden.tolist() if t is not None}):
                        cells = I[hidden == category]
                        categories[category] = (response[:, cells].sum(axis=1), response[:, cells] @ base_full[cells])
                inputs[name] = {"kept": response[:, K] - own, "base_kept": base_full[K],
                                "categories": categories, "interior": J_full[I]}
            full_group = group_of[(k, "full")]
            group_outputs = {}
            for name in reduction.output_names:
                if ds.has(name):
                    values = np.asarray(ds.get(name), dtype=float)
                    group_outputs[name] = float(values[full_group] if ds.location(name) == reduction.location else
                                                values[np.flatnonzero(owner == full_group)[0]])
            kind = Kind(D=S - A_red, e=s - c_red, reduction=kron, interior_source=c_full[I],
                        kept_rows=compartments[K], interior_rows=compartments[I], inputs=inputs,
                        outputs=group_outputs, key=kind_key, system=system)
            for name, method in derived.items():   # while the factorisation is live
                kind.outputs[name] = method(kind, full_template, kron, full_entities)
            kron.release()                     # its factorisation (the full interior's) made again on recovery
            out[kind_key][system] = kind
    return out

