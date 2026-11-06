"""
Reduction of graph systems by groups: a graph system whose nodes come in many repeated groups (sections, voxels,
motifs) solved on a reduced form. Each group of the reduced location holds only its kept nodes; its interior is
eliminated exactly, once per kind of group, from a full instance evaluated on a library; its interior's values are
recovered on request. Exact when the eliminated interior is linear in the unknowns; a nonlinear term is an
@explicit_term of the graph system, its nodes kept.

Declared on the plant model, before its structure is built (as nodes= and wiring are):

    class Plant(CompositeModel):
        reduction = Reduction(location="SubOrgan", labels=("RootSegment",), ports=conduits, keep=soil_nodes,
                              kind={"barrier": None, "length": Relative(0.02)},
                              inputs={"osmotic_potential": dict(by="tissue", group="tissue_osmotic_potential",
                                                                kept="conduit_osmotic_potential")},
                              systems=("MechaHydraulics._plant_problem",))

  location, labels   the groups (the entities of a coarse location) and the labels of those reduced (all: ())
  ports, keep        rules on a full SectionTemplate (template -> mask of its local vertices): the nodes joined across
                     groups (the structural model's) and the nodes the functional model keeps (its boundary nodes,
                     the nodes of its explicit terms)
  explicit_edges     a rule giving the Connections kept between kept nodes (those of the explicit terms)
  kind               the group variables a kind depends on, {name: None (its value) | Relative(width) (its class)};
                     the plant's parameters and the components' settings are always in it
  inputs             the per-group right-hand-side inputs {node variable: {"by": a template property (its categories
                     over the interior), "group": the group's {category: value} input, "kept": the kept nodes' input
                     (NaN: the kind's)}}, acting through the graph system's @input_jacobian(variable)
  systems            the graph systems reduced, "Component._system"
  outputs            the group variables a kind gives its groups (their values on its library's full group, e.g. a
                     section's kr): Reduction.outputs(ds, name)

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
    """Classes of relative width *width* ((1 + width)^k): a value's class is the nearest."""
    width: float

    def classify(self, value) -> float:
        value = float(value)
        if value <= 0 or not np.isfinite(value):
            return value
        base = np.log1p(float(self.width))
        return float(np.exp(np.round(np.log(value) / base) * base))


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
    """A kind's reduction: its term on the kept nodes (D, e), the inputs' responses, and its interior for recovery."""
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

    def release(self) -> None:
        """Free its arrays: a kind no group of its DataStructure has any more (Reduction.group_kinds' release), its
        memory freed even where a cache still holds it. Its derived values (@kind_derived, in its outputs) go with
        it, computed again with it."""
        self.D = self.e = self.reduction = self.interior_source = None
        self.kept_rows = self.interior_rows = None
        self.inputs, self.outputs = {}, {}


class Reduction:
    def __init__(self, location: str = "SubOrgan", labels=(), ports=None, keep=None, explicit_edges=None,
                 kind: dict = None, inputs: dict = None, systems=(), outputs=()):
        self.location, self.labels = location, tuple(labels)
        self.ports, self.keep, self.explicit_edges = ports, keep, explicit_edges
        self.kind, self.inputs, self.systems = dict(kind or {}), dict(inputs or {}), tuple(systems)
        self.output_names = tuple(outputs)
        self._reduced, self._full, self._kept, self._rows = {}, {}, {}, {}
        self._kinds = {}
        self.model, self.instance, self.g = None, None, None
        self.library = False                     # a library's: its groups' kinds are not evaluated (no library of
                                                 # its own)

    def __getstate__(self):
        state = dict(self.__dict__)                       # a checkpoint: the declaration and the templates only
        state.update(_kinds={}, model=None, instance=None, g=None)
        return state

    # ── The population's own (Scene, population_data_structure) ──

    def for_population(self, g) -> "Reduction":
        """A copy for one population's MPG *g* (the templates shared, the kinds its own)."""
        out = copy.copy(self)
        out._kinds, out.g = {}, g
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

    def _kind_values(self, ds, groups) -> dict:
        """{group: (the kind variables' values or classes)}."""
        out = {i: [] for i in groups}
        for name, classes in self.kind.items():
            values = np.asarray(ds.get(name))
            location = ds.location(name)
            owner = np.asarray(ds.owner(self.location)) if location == "node" else None
            for i in groups:
                value = float(values[np.flatnonzero(owner == i)[0]]) if owner is not None else float(values[i])
                out[i].append(value if classes is None else classes.classify(value))
        return {i: tuple(v) for i, v in out.items()}

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
        """{group: {system: Kind}}: the kinds of *groups*, the missing ones evaluated together (per plant's
        parameters). *release*: *groups* are all the DataStructure's reduced groups (a solve's terms), and the kinds
        none of them has are released: a growing section's former kinds (its length's former classes), their
        memory freed, evaluated again if a group comes back to one. So the kinds kept are those of the plant's
        current groups, not of every group it had."""
        values = self._kind_values(ds, groups)
        plant = np.asarray(ds.owner("Plant"))
        owner = np.asarray(ds.owner(self.location))
        by_plant, keys, missing = {}, {}, {}
        for i, (vertex, key) in groups.items():
            p = int(plant[np.flatnonzero(owner == i)[0]])
            if p not in by_plant:
                parameters = self._parameters(ds, p)
                by_plant[p] = (parameters, _signature(parameters))
            parameters, signature = by_plant[p]
            kind_key = (key, values[i], signature)
            keys[i] = kind_key
            if kind_key not in self._kinds:
                missing.setdefault(signature, (parameters, {}))[1].setdefault(kind_key, (vertex, key, values[i]))
        for parameters, requests in missing.values():
            self._kinds.update(_evaluate(self, parameters, requests))
        if release:
            used = set(keys.values())
            for kind_key in [k for k in self._kinds if k not in used]:
                for kind in self._kinds.pop(kind_key).values():
                    kind.release()
        return {i: self._kinds[k] for i, k in keys.items()}

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
            by_kind.setdefault(id(kind), (kind, []))[1].append((i, owner, key, values[ds.index_of(kept)]))
        out = {}
        for kind, members in by_kind.values():
            columns = np.column_stack([member[3] for member in members])
            interior = kind.reduction.recover(columns, -np.asarray(kind.interior_source, dtype=float)[:, None])
            interior = np.asarray(interior).reshape(len(kind.interior_rows), len(members))
            for column, (i, owner, key, _) in enumerate(members):
                out[i] = (owner, key, self._full[key], kind.interior_rows, interior[:, column].copy())
            del interior, columns
        return out

    def recover(self, ds, group: int, values, system: str = None) -> tuple:
        """A reduced group's interior: (its full template's local rows, the values of the system's unknown there),
        from its kind and its kept nodes' *values* (the unknown on them, DataStructure order)."""
        groups = self.reduced_groups(ds)
        kind = self.group_kinds(ds, {group: groups[group]})[group][system or self.systems[0]]
        x = np.asarray(values, dtype=float)
        return kind.interior_rows, kind.reduction.recover(x, -kind.interior_source)


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


def _evaluate(reduction: Reduction, parameters: dict, requests: dict) -> dict:
    """{kind key: {system: Kind}} of *requests* ({kind key: (a representative group vertex, its reduced template
    key, its kind values)}) for a plant of *parameters*: a library population of the plant model, one full and one
    reduced instance per kind, isolated."""
    from openalea.metafspm.coupling.component import StructuralComponent
    from openalea.metafspm.scene.population import Planting, planting, population_data_structure
    from openalea.metafspm.solve.decorator import _GraphSystemBuilder, declared_system, kind_derived_methods
    items = list(requests.items())
    plant_g = reduction.g
    scale = getattr(plant_g.scales, reduction.location)
    groups_made = {}

    class LibraryStructure(StructuralComponent):
        @classmethod
        def initiate_plant(cls, g, plant, scenario):
            from openalea.metafspm.data_structure.configs import PropsConfig
            for k, (_, (vertex, key, _)) in enumerate(items):
                chain = []                                 # the representative's complexes, Plant excluded
                v = plant_g.complex(vertex)
                while v is not None and plant_g.scale(v) > plant_g.scales.Plant:
                    chain.append(v)
                    v = plant_g.complex(v)
                for which, template in (("full", reduction._full[key]), ("reduced", reduction._reduced[
                        reduction._full[key].key])):
                    parent = plant
                    for v in reversed(chain):
                        parent = g.add_component(parent, **PropsConfig(scale=plant_g.scale(v), edge_type='/',
                                                                       label=plant_g.label(v)))
                    group = g.add_component(parent, **PropsConfig(scale=scale, edge_type='/',
                                                                  label=plant_g.label(vertex)))
                    for name, prop in plant_g.properties().items():
                        value = prop.get(vertex) if hasattr(prop, "get") else None
                        if value is not None and name not in ("scale", "edge_type", "label", "isanchor"):
                            g.property(name)[group] = value
                    g.add_instance(group, template)
                    groups_made[group] = (k, which)

    model = type(f"Library{reduction.model.__name__}", (reduction.model,),
                 dict(initiators=(LibraryStructure,), wiring=(), reduction=None))
    from openalea.metafspm.coupling.choregrapher import Choregrapher
    time_step = Choregrapher().simulation_step()             # the scene's step, kept for its components after
    population = population_data_structure(planting([Planting()], model, {}), model,
                                           time_step=3600. if time_step is None else time_step)
    ds = population.data_structure
    instance = population.instance
    library = reduction.for_population(ds.structure)       # the library knows its reduced groups, evaluates none
    library.library = True
    library.bind(model, instance)
    ds.structure.reduction = library
    from dataclasses import fields
    for component in instance.components:                    # the plant's parameters and the settings
        own = {f.name for f in fields(component) if f.metadata.get("variable_type") in ("setting", "parameter")}
        for name, value in parameters.items():
            if name not in own:
                continue
            if ds.has(name) and ds.location(name) == "Plant":   # the library's one plant: its value there
                stored = np.asarray(ds.get(name))
                values = np.empty(stored.shape, dtype=object) if stored.dtype == object else stored.copy()
                for i in range(len(values)):
                    values[i] = value                         # a record as it is (a dict, nested lists)
                ds.set(name, values)
            else:
                setattr(component, name, value)
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

