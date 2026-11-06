"""
Reduction of graph systems by groups: a graph system whose nodes come in many repeated groups (sections, voxels,
motifs) solved on a reduced form. Each group of the reduced location holds only its kept nodes; its interior is
eliminated exactly, once per kind of group, from a full instance evaluated on a library; its interior's values are
recovered on request. Exact when the eliminated interior is linear in the unknowns; a nonlinear term is an
@explicit_term of the graph system, its nodes kept.

Declared on the plant model, before its structure is built (as nodes= and wiring are):

    class Plant(CompositeModel):
        reduction = Reduction(location="SubOrgan", where={"label": "RootSegment"}, ports="GranapAnatomy.conduits",
                              keep=("GranapAnatomy.soil_nodes", vapour_path),
                              kind={"barrier": None, "length": Relative(0.1)},
                              inputs={"osmotic_potential": dict(by="tissue", group="tissue_osmotic_potential",
                                                                kept="conduit_osmotic_potential")},
                              systems=("MechaHydraulics._plant_problem",))

  location, where    the groups (the entities of a coarse location) and those reduced: a filter on them (the
                     entities' syntax, {variable: condition}; "label" their label names, "<Scale>.label" their
                     complex's at that scale, e.g. {"label": "StemElement", "Organ.label": "StemInternode"}: the
                     internodes' elements, not the sheaths'), or a named entity set ("Class.set"); None: all
  ports, keep        rules on a full SectionTemplate: the nodes joined across groups (the structural model's) and the
                     nodes the functional model keeps (its boundary nodes, the nodes of its explicit terms); a rule
                     is a published entity set's name ("GranapAnatomy.conduits", evaluated on the template's
                     properties), a {variable: condition} dict, a callable template -> mask of its local vertices, or
                     a tuple of them (their union)
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
the kinds are checkpointed (shared parts, saved once), so a resumed run evaluates none again. The groups are arrays
(group_table, cached per topology; assignment: each group's corners, their weights and kinds), batched by template
and kind: the solve's terms fill a sparse pattern cached per topology and node set, the outputs (outputs,
kept_outputs) and the recoveries are the kinds' values weighted; no Python loop over the groups.

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
        return float(self._nearest(np.array([float(value)]))[0])

    def corners(self, value) -> tuple:
        """((weight, class), ...) of *value*: its two neighbouring classes weighted by its place between them (one at a
        class value), or its nearest class without interpolation (a value not positive or not finite: itself)."""
        options = self.corner_arrays(np.array([float(value)]))
        return tuple((float(weight[0]), float(classes[0])) for weight, classes in options if weight[0] > 0)

    def _nearest(self, values) -> np.ndarray:
        special = ~(values > 0) | ~np.isfinite(values)
        base = np.log1p(float(self.width))
        nearest = np.exp(np.round(np.log(np.where(special, 1., values)) / base) * base)
        return np.where(special, values, nearest)

    def corner_arrays(self, values) -> tuple:
        """corners of each of *values* at once: ((weights, classes),) without interpolation, else ((low weights, low
        classes), (high weights, high classes)), the high weight 0 (and its class the low one) for a value at a class
        or not positive or not finite (corners' one class)."""
        values = np.asarray(values, dtype=float)
        if not self.interpolate:
            return ((np.ones(values.shape), self._nearest(values)),)
        special = ~(values > 0) | ~np.isfinite(values)
        safe = np.where(special, 1., values)
        base = np.log1p(float(self.width))
        k = np.floor(np.log(safe) / base)
        low, high = np.exp(k * base), np.exp((k + 1) * base)
        weight = (safe - low) / (high - low)
        at_low = weight <= CLASS_ROUNDING
        at_high = ~at_low & (weight >= 1. - CLASS_ROUNDING)
        single = special | at_low | at_high
        first = np.where(special, values, np.where(at_high, high, low))
        return ((np.where(single, 1., 1. - weight), first),
                (np.where(single, 0., weight), np.where(single, first, high)))


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
    def __init__(self, location: str = "SubOrgan", where=None, ports=None, keep=None, explicit_edges=None,
                 kind: dict = None, inputs: dict = None, systems=(), outputs=()):
        if where is not None and not isinstance(where, (dict, str)):
            raise TypeError(f"Reduction: where is a {{variable: condition}} filter of the {location}s or an entity "
                            f"set's name, got {where!r}")
        self.location, self.where = location, (dict(where) if isinstance(where, dict) else where)
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
        state.pop("_table", None)          # the groups' arrays: per DataStructure and topology
        state.pop("_assigned", None)
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
        out.__dict__.pop("_table", None)
        out.__dict__.pop("_assigned", None)
        return out

    def bind(self, model, instance) -> None:
        """The plant model (its class, for the library) and its instance (its components' parameters)."""
        self.model, self.instance = model, instance

    # ── Structure ──

    def reduces(self, g, vertex) -> bool:
        """Whether the group *vertex* of the MPG *g* is reduced: where= holds for it (its label, its complexes'
        labels, its properties)."""
        if self.where is None:
            return True
        from openalea.metafspm.coupling.declaration import entity_set, rule_holds
        rule = entity_set(self.where) if isinstance(self.where, str) else self.where
        values, scales = self._vertex_values(g, int(vertex), rule)

        def codes(key, condition):                  # label names -> the codes of their scale
            if key not in scales:
                return condition
            names = vars(getattr(g.labels, scales[key]))
            named = list(condition) if isinstance(condition, (list, tuple, set)) else [condition]
            unknown = [name for name in named if isinstance(name, str) and name not in names]
            if unknown:
                raise KeyError(f"Reduction where: {unknown} are no {scales[key]} labels")
            coded = [names[name] if isinstance(name, str) else name for name in named]
            return coded if isinstance(condition, (list, tuple, set)) else coded[0]

        return bool(rule_holds(rule, values, codes)[0])

    @staticmethod
    def _vertex_values(g, vertex: int, rule: dict) -> tuple:
        """({key: [the vertex's value]}, {label key: its scale's name}) for the keys of *rule*: "label" its label
        code, "<Scale>.label" its complex's at that scale, another its property's (NaN without one)."""
        from openalea.metafspm.coupling.declaration import scale_names
        names = scale_names(g)
        values, scales = {}, {}
        labels = g.property("label")
        for key in rule:
            if key == "label" or key.endswith(".label"):
                at = vertex
                if key != "label":
                    scale = getattr(g.scales, key[:-len(".label")], None)
                    if scale is None:
                        raise KeyError(f"Reduction where: no scale '{key[:-len('.label')]}'")
                    at = g.complex_at_scale(vertex, scale)
                scales[key] = names[g.scale(at)]
                values[key] = np.array([float(labels.get(at, np.nan))])
            else:
                value = g.property(key).get(vertex, np.nan) if key in g.properties() else np.nan
                values[key] = np.array([value], dtype=float)
        return values, scales

    def labels(self) -> tuple:
        """The label names where= names (its "label" filter), () when it names none: e.g. for a port's warning of a
        label it has no groups of."""
        rule = self.where if isinstance(self.where, dict) else {}
        label = rule.get("label", ())
        return (label,) if isinstance(label, str) else tuple(label)

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
        from openalea.metafspm.coupling.declaration import template_selection
        kept = np.zeros(full.size, dtype=bool)
        for rule in (self.ports, self.keep):
            if rule is not None:
                kept |= template_selection(rule, full)
        kept &= compartment
        structure = np.zeros(full.size, dtype=bool)
        above = full.complex[kept & (full.complex >= 0)]
        while above.size:
            structure[above] = True
            above = full.complex[above]
            above = above[above >= 0]
        edges = np.zeros(full.size, dtype=bool)
        if self.explicit_edges is not None:
            edges = template_selection(self.explicit_edges, full) & (full.scale == ScalesConfig.Connection)
        return kept, kept | structure | edges

    def is_reduced(self, template) -> bool:
        """Whether *template* (a SectionTemplate) is one of this reduction's reduced templates (template_for's)."""
        return template.key in self._full

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

    def group_table(self, ds):
        """The DataStructure's reduced groups as arrays (_GroupTable: their entity indices, vertices, templates, first
        nodes, plants, kept nodes), None without any; cached per topology (read-only arrays)."""
        stamp = ds._topology_stamp() if hasattr(ds, "_topology_stamp") else ds.topology_version
        cached = self.__dict__.get("_table")
        if cached is not None and cached[0] == stamp and cached[1] == id(ds):
            return cached[2]
        table = _GroupTable.build(self, ds)
        self.__dict__["_table"] = (stamp, id(ds), table)
        return table

    def reduced_groups(self, ds) -> dict:
        """{group entity index: (its vertex, its reduced template key)} of the DataStructure's reduced groups."""
        table = self.group_table(ds)
        if table is None:
            return {}
        return dict(zip(table.index.tolist(), zip(table.owner.tolist(), table.template)))

    # ── The terms of a graph system's solve ──

    def terms(self, component, system: str, ds, node_ids) -> tuple:
        """(D, e) on the solved nodes *node_ids*: the reduced groups' interiors' terms (sparse, nodes × nodes, and a
        vector), or None when no solved group is reduced. Batched by template and kind: the sparse pattern is cached
        per topology and node set, only its values (each group's kinds' weighted D) filled at each call."""
        if self.library:
            return None
        assignment = self.assignment(ds, release=True)
        if assignment is None:
            return None
        pattern = assignment.table.pattern(node_ids)
        if pattern is None:
            return None
        n = pattern.size
        values, e = [], np.zeros(n)
        for rows, local in pattern.blocks:
            D = assignment.combine(rows, lambda kind: kind[system].D)
            vector = assignment.combine(rows, lambda kind: kind[system].e)
            vector = vector + self._input_terms(ds, assignment, system, rows)
            e[local.reshape(-1)] += vector.reshape(-1)
            values.append(D.reshape(-1))
        data = np.concatenate(values)[pattern.permutation]
        return csr_matrix((data, pattern.indices, pattern.indptr), shape=(n, n)), e

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
        """Group variable *name* of each group of the location: its kind's (its library full group's value; a blend's
        its parts' weighted) for a reduced group, NaN elsewhere."""
        out = np.full(len(ds.entity_ids(self.location)), np.nan)
        assignment = None if self.library else self.assignment(ds)
        if assignment is None:
            return out
        rows = np.arange(assignment.table.index.size)

        def value(kind):
            found = (next(iter(kind.values())).outputs or {}).get(name, np.nan)
            try:
                return float(found)
            except (TypeError, ValueError):
                return np.nan
        out[assignment.table.index] = assignment.combine(rows, value)
        return out

    def kept_outputs(self, ds, name: str, system: str = None) -> list:
        """[(kept nodes, values), ...] per reduced template: a kind's output *name* of one value per kept node (a
        @kind_derived vector, e.g. the condensed volumes) on each reduced group's kept nodes (DataStructure indices,
        groups × kept), its kinds' weighted; [] without a reduced group."""
        assignment = None if self.library else self.assignment(ds)
        if assignment is None:
            return []
        system = system or self.systems[0]
        table = assignment.table
        return [(table.kept_of(rows), assignment.combine(rows, lambda kind: kind[system].outputs[name]))
                for rows in table.by_template]

    def _reduced_scale(self, key):
        return self._reduced[self._full[key].key].scale

    def _input_terms(self, ds, assignment, system: str, rows) -> np.ndarray:
        """The change of the groups *rows* (table rows of one template)'s kept sources for their own inputs, their
        kinds' weighted (groups × kept): the kept nodes' (kept input, NaN: the kind's) and their categories'
        ({category: value} group input, read per group: a record each)."""
        table = assignment.table
        kept = table.kept_of(rows)
        out = np.zeros(kept.shape)
        if not self.inputs:
            return out
        for corner in range(assignment.kind.shape[1]):
            chosen = assignment.kind[rows, corner]
            for k in np.unique(chosen[chosen >= 0]).tolist():
                at = np.flatnonzero(chosen == k)
                weight = assignment.weights[rows[at], corner][:, None]
                out[at] += weight * self._part_input_terms(ds, assignment.kinds[k][system], table.index[rows[at]],
                                                           kept[at])
        return out

    def _part_input_terms(self, ds, kind, groups, kept) -> np.ndarray:
        """One (not blended) kind's input terms for its *groups* (entity indices) of *kept* nodes (DataStructure
        indices, groups × kept)."""
        out = np.zeros(kept.shape)
        for name, response in kind.inputs.items():
            declared = self.inputs[name]
            if declared.get("kept") and ds.has(declared["kept"]):
                values = np.asarray(ds.get(declared["kept"]), dtype=float)[kept]
                delta = np.where(np.isfinite(values), values - response["base_kept"], 0.)
                out += delta @ np.asarray(response["kept"]).T
            if declared.get("group") and ds.has(declared["group"]):
                given = np.asarray(ds.get(declared["group"]), dtype=object).reshape(-1)
                for row, group in enumerate(groups.tolist()):
                    chosen = given[group] if group < given.size else None
                    for category, value in (chosen or {}).items():
                        if category in response["categories"]:
                            unit, base = response["categories"][category]
                            out[row] += float(value) * unit - base
        return out

    # ── Kinds ──

    def _corners(self, ds, table) -> tuple:
        """(weights, values): each group's corners (groups × corners) and their kind variables' values or classes
        (groups × corners × variables): one kind's values, or the corners of the classes it lies between
        (Relative(interpolate=True): two per such variable, their weights' products; a corner of weight 0 pads a
        group of fewer corners)."""
        size = table.index.size
        weights, values = np.ones((size, 1)), np.zeros((size, 1, 0))
        for name, classes in self.kind.items():
            raw = np.asarray(ds.get(name))
            picked = np.asarray(raw[table.first_node] if ds.location(name) == "node" else raw[table.index],
                                dtype=float)
            options = ((np.ones(size), picked),) if classes is None else classes.corner_arrays(picked)
            option_weights = np.stack([weight for weight, _ in options], axis=1)
            option_values = np.stack([value for _, value in options], axis=1)
            corners, count = weights.shape[1], option_weights.shape[1]
            weights = (weights[:, :, None] * option_weights[:, None, :]).reshape(size, corners * count)
            values = np.concatenate([np.repeat(values, count, axis=1),
                                     np.tile(option_values, (1, corners))[:, :, None]], axis=2)
        return weights, values

    def _parameters(self, ds, plant_index) -> dict:
        """The plant's parameters and the components' settings ({name: value}), for its kinds' key and library."""
        return self._plant_parameters(ds)[plant_index]

    def _plant_parameters(self, ds) -> dict:
        """{plant index: its parameters and the components' settings}, for every plant of *ds*: the fields read once
        per component class, each parameter's values once for all the plants."""
        plants = len(ds.entity_ids("Plant"))
        out = {p: {} for p in range(plants)}
        for component in self.instance.components:
            settings, numeric, located, records = _parameter_fields(type(component))
            for name in settings:
                value = getattr(component, name)
                for p in out:
                    out[p][name] = value
            for name in numeric:
                if name in located:          # per entity (an edge's, a node's): the DataStructure's, not a kind's
                    continue
                try:
                    values = np.asarray(component.parameter_values(name, "Plant"), dtype=float).reshape(-1)
                except (KeyError, ValueError, AttributeError):
                    values = np.asarray([getattr(component, name)], dtype=float)
                for p in out:
                    out[p][name] = float(values[p] if values.size > p else values[0])
            for name in records:             # its Plant-scale parameters, any value (a dict, a boolean)
                if not (ds.has(name) and ds.location(name) == "Plant"):
                    continue
                values = ds.get(name)
                for p in out:
                    if name in out[p]:
                        continue
                    value = values[p] if len(values) > p else values[0]
                    out[p][name] = value.item() if isinstance(value, np.generic) else value
        return out

    def assignment(self, ds, release: bool = False):
        """
        The kinds of the DataStructure's reduced groups (_Assignment: each group's corners, their weights and kinds),
        None without a reduced group; the missing kinds evaluated (per plant's parameters, in batches:
        _evaluate_missing). Vectorised: the groups' corners as arrays, a loop over the distinct kinds only. *release*:
        the kinds no group has are released (their memory freed, evaluated again if a group comes back to one),
        except a class's whose exact kind values, template and parameters a current group has (a growing section's
        former classes: kept, a bounded set, so that the kinds saturate). Without classes the kinds kept are those of
        the plant's current groups only.
        """
        table = self.group_table(ds)
        if table is None:
            return None
        parameters = self._plant_parameters(ds)
        signatures = {p: _signature(parameters[p]) for p in np.unique(table.plant).tolist()}
        stamp = (id(table), tuple(ds.write_count(name) for name in self.kind), tuple(sorted(signatures.items())))
        cached = self.__dict__.get("_assigned")
        if (cached is not None and cached[0] == stamp
                and all(self._kinds.get(key) is kind for key, kind in zip(cached[1].keys, cached[1].kinds))):
            if release:                    # the same groups, values and parameters as the last call: its kinds
                self._release(set(cached[1].keys))
            return cached[1]
        weights, values = self._corners(ds, table)
        signature_names = sorted(set(signatures.values()))
        lookup = np.zeros(int(table.plant.max()) + 1 if table.plant.size else 0, dtype=np.int64)
        for p, signature in signatures.items():
            lookup[p] = signature_names.index(signature)
        signature_code = lookup[table.plant]
        group, corner = np.nonzero(weights > 0)          # group by group, corner by corner (their declared order)
        chosen = np.ascontiguousarray(values[group, corner])
        rows = np.column_stack([table.code[group], signature_code[group], chosen.view(np.int64)])
        _, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
        keys, position, local = [], {}, np.empty(first.size, dtype=np.int64)
        missing = {}
        for u in np.argsort(first, kind="stable").tolist():                 # in order of first appearance
            at = int(first[u])
            g = int(group[at])
            kind_values = tuple(float(value) for value in chosen[at])
            signature = signature_names[int(signature_code[g])]
            key = (table.keys[int(table.code[g])], kind_values, signature)
            if key not in position:                      # -0. and 0.: one key
                position[key] = len(keys)
                keys.append(key)
                if key not in self._kinds:
                    plant = int(table.plant[g])
                    missing.setdefault(signature, (parameters[plant], {}))[1].setdefault(
                        key, (int(table.owner[g]), key[0], kind_values))
            local[u] = position[key]
        for plant_parameters, requests in missing.values():
            self._evaluate_missing(plant_parameters, requests)
        kind = np.full(weights.shape, -1, dtype=np.int64)
        kind[group, corner] = local[np.asarray(inverse).reshape(-1)]
        out = _Assignment(table=table, weights=weights, kind=kind, keys=keys,
                          kinds=[self._kinds[key] for key in keys])
        if release:
            self._release(set(keys))
        self.__dict__["_assigned"] = (stamp, out)
        return out

    def _release(self, used: set) -> None:
        """The kinds not in *used* released, a class's of a current family kept (assignment's rule); the blends
        dropped (made again when asked)."""
        exact = [position for position, classes in enumerate(self.kind.values()) if classes is None]
        current = {self._class_family(kind_key, exact) for kind_key in used}
        for kind_key in [k for k in self._kinds if k not in used]:
            if len(exact) < len(self.kind) and self._class_family(kind_key, exact) in current:
                continue                              # a class of a current group's family: kept
            for kind in self._kinds.pop(kind_key).values():
                kind.release()
        self.__dict__["_blends"] = {}

    def group_kinds(self, ds, groups, release: bool = False) -> dict:
        """
        {group: {system: Kind}}: the kinds of *groups* (entity indices, e.g. reduced_groups' keys), the missing ones
        evaluated (assignment's); a group between two classes of a Relative variable gets their blend (_blend, kept
        until the next release). *release*: as assignment's (*groups* all the DataStructure's reduced groups). For
        a group's own kind; the solve's terms, the outputs and the recoveries use assignment's arrays.
        """
        assignment = self.assignment(ds, release=release)
        if assignment is None:
            return {}
        table = assignment.table
        row_of = dict(zip(table.index.tolist(), range(table.index.size)))
        blends = self.__dict__.setdefault("_blends", {})
        out = {}
        for i in groups:
            row = row_of[int(i)]
            on = np.flatnonzero(assignment.kind[row] >= 0)
            chosen = tuple((float(assignment.weights[row, c]), assignment.keys[assignment.kind[row, c]]) for c in on)
            if len(chosen) == 1:
                out[i] = self._kinds[chosen[0][1]]
                continue
            if chosen not in blends:
                blends[chosen] = {system: _blend([(weight, self._kinds[kind_key][system])
                                                  for weight, kind_key in chosen])
                                  for system in self._kinds[chosen[0][1]]}
            out[i] = blends[chosen]
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
        assignment = None if self.library else self.assignment(ds)
        if assignment is None:
            return {}
        table = assignment.table
        if groups is None:
            rows = np.arange(table.index.size)
        else:
            row_of = np.full(int(max(table.index.max(), max((int(i) for i in groups), default=0))) + 1, -1,
                             dtype=np.int64)
            row_of[table.index] = np.arange(table.index.size)
            asked = np.asarray([int(i) for i in groups], dtype=np.int64)
            rows = row_of[asked[(asked >= 0) & (asked < row_of.size)]]
            rows = np.unique(rows[rows >= 0])
        system = system or self.systems[0]
        values = np.asarray(ds.get(self.unknown(system)), dtype=float)
        out = {}
        for code_rows in table.by_template:
            at = code_rows[np.isin(code_rows, rows)]
            if not at.size:
                continue
            columns = values[table.kept_of(at)]                          # groups × kept
            chosen = assignment.kind[at]
            for k in np.unique(chosen[chosen >= 0]).tolist():          # each kind solved once, its groups at once
                part = assignment.kinds[k][system]
                members, corners = np.nonzero(chosen == k)
                interior = np.asarray(part.recover(columns[members].T)).reshape(len(part.interior_rows),
                                                                               members.size)
                weights = assignment.weights[at[members], corners]
                for column, (member, weight) in enumerate(zip(members.tolist(), weights.tolist())):
                    row = int(at[member])
                    i = int(table.index[row])
                    if i in out:
                        out[i][4][...] += weight * interior[:, column]
                    else:
                        key = table.template[row]
                        out[i] = (int(table.owner[row]), key, self._full[key], part.interior_rows,
                                  weight * interior[:, column])
                del interior
        return dict(sorted(out.items()))

    def recover(self, ds, group: int, values, system: str = None) -> tuple:
        """A reduced group's interior: (its full template's local rows, the values of the system's unknown there),
        from its kind and its kept nodes' *values* (the unknown on them, DataStructure order)."""
        kind = self.group_kinds(ds, (group,))[group][system or self.systems[0]]
        return kind.interior_rows, kind.recover(values)


class _GroupTable:
    """
    A DataStructure's reduced groups as arrays, in their instances' order (Reduction.group_table, cached per
    topology): their entity indices at the location (index), vertices (owner), reduced template keys (template, and
    code: the position in keys), first vertex (first), first node (first_node, a DataStructure index), plant (an
    entity index); by template, the table rows of each (by_template) and their kept nodes' vertices (kept_vids,
    rows × kept). Every array read-only.
    """

    def __init__(self, ds, index, owner, template, keys, code, first, first_node, plant, offsets):
        self.ds_id = id(ds)
        self.index, self.owner, self.template, self.keys, self.code = index, owner, template, keys, code
        self.first, self.first_node, self.plant = first, first_node, plant
        self.by_template = [np.flatnonzero(code == c) for c in range(len(keys))]
        self.within = np.empty(code.size, dtype=np.int64)          # each row's position in its template's rows
        for rows in self.by_template:
            self.within[rows] = np.arange(rows.size)
        self.kept_vids = [first[rows][:, None] + offsets[c][None, :] for c, rows in enumerate(self.by_template)]
        self._kept_nodes = [None] * len(keys)
        self._index_of = ds.index_of
        self._patterns = {}
        for array in (index, owner, code, first, first_node, plant, self.within, *self.by_template, *self.kept_vids):
            array.setflags(write=False)

    @classmethod
    def build(cls, reduction, ds):
        g = ds.structure
        if not g._instanced():
            return None
        instances = g.instances
        code_of = {code: key for key, code in instances.codes.items()}
        reduced = np.array([code for key, code in instances.codes.items() if key in reduction._full], dtype=np.int64)
        rows = np.flatnonzero(np.isin(instances.code, reduced))
        ids = np.asarray(ds.entity_ids(reduction.location), dtype=np.int64)
        owner = instances.owner[rows]
        position = np.searchsorted(ids, owner).clip(0, max(ids.size - 1, 0))
        located = (ids.size > 0) & (ids[position] == owner) if ids.size else np.zeros(owner.size, dtype=bool)
        rows, owner, index = rows[located], owner[located], position[located]
        if not rows.size:
            return None
        instance_code = instances.code[rows]
        used, code = np.unique(instance_code, return_inverse=True)
        keys = [code_of[int(c)] for c in used.tolist()]
        template = [keys[c] for c in code.tolist()]
        from openalea.metafspm.data_structure.configs import ScalesConfig
        offsets = [np.flatnonzero(reduction._reduced_scale(key) == ScalesConfig.Compartment) for key in keys]
        location_owner = np.asarray(ds.owner(reduction.location))
        first_node = _first_nodes(location_owner)
        first_node = np.where(index < first_node.size, first_node[np.minimum(index, max(first_node.size - 1, 0))],
                              -1) if first_node.size else np.full(index.size, -1, dtype=np.int64)
        plant_of = np.asarray(ds.owner("Plant"))
        plant = np.where(first_node >= 0, plant_of[np.maximum(first_node, 0)], -1) if plant_of.size else \
            np.full(index.size, -1, dtype=np.int64)
        return cls(ds, index.astype(np.int64), owner.astype(np.int64), template, keys,
                   np.asarray(code, dtype=np.int64).reshape(-1), instances.first[rows].astype(np.int64),
                   first_node.astype(np.int64), plant.astype(np.int64), offsets)

    def kept_of(self, rows) -> np.ndarray:
        """The kept nodes (DataStructure indices, rows × kept) of the table *rows*, all of one template."""
        rows = np.asarray(rows, dtype=np.int64)
        if not rows.size:
            return np.zeros((0, 0), dtype=np.int64)
        c = int(self.code[rows[0]])
        if self._kept_nodes[c] is None:
            vids = self.kept_vids[c]
            nodes = np.asarray(self._index_of(vids.reshape(-1)), dtype=np.int64).reshape(vids.shape)
            nodes.setflags(write=False)
            self._kept_nodes[c] = nodes
        return self._kept_nodes[c][self.within[rows]]

    def pattern(self, node_ids):
        """The sparse pattern of the groups' terms on the solved nodes *node_ids* (_Pattern), None when none of the
        groups is solved (all its kept nodes among them); cached per node set."""
        node_ids = np.asarray(node_ids, dtype=np.int64)
        key = (node_ids.size, hash(node_ids.tobytes()))
        if key not in self._patterns:
            self._patterns.clear()                       # one node set at a time (a step's solve)
            self._patterns[key] = _Pattern.build(self, node_ids)
        return self._patterns[key]


class _Pattern:
    """The COO pattern of the reduced groups' terms on a node set, as CSR: blocks [(table rows, their kept nodes'
    positions: rows × kept)], one per template; permutation: the CSR order of the blocks' entries (each group's
    kept × kept, row by row), indices and indptr."""

    def __init__(self, size, blocks, permutation, indices, indptr):
        self.size, self.blocks = size, blocks
        self.permutation, self.indices, self.indptr = permutation, indices, indptr

    @classmethod
    def build(cls, table, node_ids):
        n = node_ids.size
        order = np.argsort(node_ids, kind="stable")
        sorted_ids = node_ids[order]
        blocks, rows_all, columns_all = [], [], []
        for rows, vids in zip(table.by_template, table.kept_vids):
            if not rows.size:
                continue
            position = np.searchsorted(sorted_ids, vids).clip(0, max(n - 1, 0))
            found = (sorted_ids[position] == vids) if n else np.zeros(vids.shape, dtype=bool)
            solved = found.all(axis=1)                      # a group not solved (a filter) left out
            if not solved.any():
                continue
            local = order[position[solved]]
            local.setflags(write=False)
            blocks.append((rows[solved], local))
            k = local.shape[1]
            rows_all.append(np.repeat(local, k, axis=1).reshape(-1))
            columns_all.append(np.tile(local, (1, k)).reshape(-1))
        if not blocks:
            return None
        rows_all, columns_all = np.concatenate(rows_all), np.concatenate(columns_all)
        entries = np.arange(1, rows_all.size + 1, dtype=np.float64)       # 1-based: no explicit zero dropped
        csr = coo_matrix((entries, (rows_all, columns_all)), shape=(n, n)).tocsr()
        permutation = csr.data.astype(np.int64) - 1
        return cls(n, blocks, permutation, csr.indices.copy(), csr.indptr.copy())


@dataclass
class _Assignment:
    """The kinds of a DataStructure's reduced groups (Reduction.assignment): table (its _GroupTable); weights and
    kind (table rows × corners: each corner's weight and its kind's position in keys and kinds, -1 for a padding
    corner of weight 0); keys and kinds ({system: Kind}) the distinct kinds."""
    table: _GroupTable
    weights: np.ndarray
    kind: np.ndarray
    keys: list
    kinds: list

    def combine(self, rows, value_of) -> np.ndarray:
        """Each of the table *rows*' value: its corners' kinds' value_of(kind) weighted ((rows,) + the values'
        shape), in the corners' order (a single kind's value as it is); a loop over the distinct kinds only."""
        rows = np.asarray(rows, dtype=np.int64)
        chosen = self.kind[rows]
        used = np.unique(chosen[chosen >= 0])
        if not used.size:
            return np.zeros(rows.size)
        stacked = np.stack([np.asarray(value_of(self.kinds[k]), dtype=float) for k in used.tolist()])
        out = np.zeros((rows.size,) + stacked.shape[1:])
        trailing = (1,) * (stacked.ndim - 1)
        for corner in range(chosen.shape[1]):
            on = np.flatnonzero(chosen[:, corner] >= 0)
            if not on.size:
                continue
            weight = self.weights[rows[on], corner].reshape((-1,) + trailing)
            out[on] += weight * stacked[np.searchsorted(used, chosen[on, corner])]
        return out


_FIELDS = {}


def _parameter_fields(component_class) -> tuple:
    """(settings, Plant parameters, those located per entity, Plant-scale parameters of any value) of a component
    class: its fields read once (Reduction._plant_parameters)."""
    if component_class not in _FIELDS:
        from dataclasses import fields
        declared = fields(component_class)
        settings = tuple(f.name for f in declared if f.metadata.get("variable_type") == "setting")
        located = frozenset(f.name for f in declared
                            if f.metadata.get("scale") is not None or f.metadata.get("location") is not None)
        records = tuple(f.name for f in declared if f.metadata.get("variable_type") == "parameter"
                        and f.metadata.get("scale") == _plant_scale())
        _FIELDS[component_class] = (settings, tuple(component_class._plant_parameter_fields()), located, records)
    return _FIELDS[component_class]


def _first_nodes(owner) -> np.ndarray:
    """Each group's first node (its index among the nodes; -1 for a group without any), from the nodes' *owner*
    groups (entity indices)."""
    owner = np.asarray(owner, dtype=np.int64)
    out = np.full(int(owner.max()) + 1 if owner.size else 0, -1, dtype=np.int64)
    owned = owner >= 0
    groups, starts = np.unique(owner[owned], return_index=True)
    out[groups] = np.flatnonzero(owned)[starts]
    return out


_PICKLES = {}            # the plants' parameter objects' pickles (content_key's): pickled once, not at every solve
PICKLES_KEPT = 4096      # at most (the dict emptied beyond)


def _signature(parameters: dict) -> str:
    """A digest of a plant's parameters, by content (dicts, records and numbers alike)."""
    from openalea.metafspm.solve.memo import content_key
    if len(_PICKLES) > PICKLES_KEPT:
        _PICKLES.clear()
    return content_key({name: parameters[name] for name in sorted(parameters)}, pickles=_PICKLES)


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
    steps = [component.simulation_step() for component in getattr(reduction.instance, "components", ())
             if hasattr(component, "simulation_step")]
    time_step = next((float(step) for step in steps if step is not None), 3600.)    # the plants' DataStructure's
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

