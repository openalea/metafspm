"""
metafspm in an fspm-TF coupling session (openalea.fspm_tf.coupling, specification §11): another platform's model
as a metafspm component.

A RemoteComponent computes nothing. Its state variables are the remote model's outputs, read from the session at
each turn; its inputs are metafspm's outputs, sent to the session. A subclass declares them as any component does
(state_variable, input_variable: unit, kind, scale) and says what metafspm does with the shared structure::

    @dataclass
    class RemoteCarbon(RemoteComponent):
        participant = "metafspm"                       # this model's name in the session
        rights = {"Plant": "owns", "Axis": "owns", "GrowthUnit": "owns", "Phytomer": "owns", "Organ": "owns",
                  "SubOrgan": "owns"}                  # its own components grow these scales
        links = [Link("metafspm", "C_hexose_root", "carbon", "hexose", aggregation="mean")]
        C_hexose_root: float = state_variable(unit="mol.g-1", scale=scales.SubOrgan, kind="massic_concentration")
        length: float = input_variable(unit="m", scale=scales.SubOrgan)

    growth, remote = RootGrowth(ds), RemoteCarbon(ds)
    remote.start()                                     # join the session: the initial state exchanged
    for _ in range(steps):
        growth()
        remote()                                       # this step's outputs sent, the next turn's inputs applied

Rights per scale: "owns" and "refines", metafspm's own components create (and remove) the scale's entities, which
are sent to the session; "follows", the entities the other participants create are created in the MPG (under their
complex, after their parent), and removed when they are. When remote entities appear, metafspm shares its own
variables over them by its repartition rules (StructuralComponent's), with *partition_weight* (a variable the
remote model provides, e.g. a length): an extensive quantity split, a concentration diluted.

Ids: the session id of a vertex metafspm created is its vid plus the participant's offset; the vertices created for
remote entities keep the remote ids in a map.
"""
from dataclasses import dataclass, fields

import numpy as np

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.io.fspm_tf import _KINDS, _location_scales, _scale_names, to_frame, ucum_unit

_EDGE_SYMBOLS = {"successor": "<", "branch": "+"}
_NO_ID = -1


def _coupling():
    try:
        from openalea.fspm_tf import coupling
    except ImportError:
        raise ImportError("openalea.fspm_tf (with its coupling) is not installed: pip install -e ../fspm-TF") from None
    return coupling


@dataclass
class RemoteComponent(StructuralComponent):
    """Another platform's model in a coupling session (see the module's documentation)."""

    participant = "metafspm"            # this metafspm model's name in the session
    data_structure_id = "plant"         # the session's data structure this DataStructure is
    platform = "metafspm"
    rights = {}                         # {scale name: "owns" | "refines" | "follows"}
    links = ()                          # the links proposed (fspm_tf.coupling.Link)
    endpoint = None                     # the session (default: the environment's FSPM_TF_ENDPOINT)
    timeout = 600.                      # seconds waited for a turn

    # ── what it declares ───────────────────────────────────────────────────────────
    def _variables(self) -> dict:
        """{name: (role, metadata)} of its state variables and inputs."""
        return {item.name: (item.metadata.get("variable_type"), dict(item.metadata)) for item in fields(type(self))
                if item.metadata.get("variable_type") in ("state_variable", "input")}

    def _set_name(self, name: str) -> str:
        """The entity set (scale name) of variable *name*'s location."""
        ds = self.data_structure
        return _scale_names(ds.structure)[_location_scales(ds)[ds.location(name)]]

    def declaration(self):
        coupling = _coupling()
        from openalea.fspm_tf import vocabulary
        standard_names = {key: value for key, value in vocabulary.platform_names("metafspm").items() if value}
        declaration = coupling.Declaration(self.participant, self.platform)
        for name, rights in self.rights.items():
            declaration.entity_sets.append(coupling.EntitySetSpec(self.data_structure_id, name, rights=rights))
        for name, (role, meta) in self._variables().items():
            declaration.variables.append(coupling.VariableSpec(
                self.data_structure_id, self._set_name(name), name,
                "requires" if role == "state_variable" else "provides", ucum_unit(meta.get("unit")),
                standard_names.get(name), _KINDS.get(meta.get("kind")), optional=False))
        return declaration.complete()

    # ── the session ────────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Join the session: the initial turn's entities and inputs applied, metafspm's initial state sent, the first
        step's inputs applied."""
        coupling = _coupling()
        self._sent = {}                           # entity set: the session ids of metafspm's entities sent last
        self._vid_of, self._session_of = {}, {}   # remote session id <-> vid of the vertex created for it
        self._echo = {}                           # remote session id: the topology it was received with
        self.stopped = None
        declaration = self.declaration()
        self._declared = {entity_set.name for entity_set in declaration.entity_sets}
        self.session = coupling.Participant.connect(declaration, self.endpoint, links=self.links,
                                                    timeout=self.timeout)
        self._turns = self.session.turns()
        self._apply_next()
        self._send()
        self._apply_next()

    def __call__(self):
        """This step's outputs and topology sent; the next turn's applied (or the session's stop noted)."""
        if self.stopped is not None:
            raise RuntimeError(f"{type(self).__name__}: the session has stopped ({self.stopped})")
        self._send()
        self._apply_next()

    def _session_id(self, vid: int) -> int:
        return self._session_of.get(vid, self.session.id_offset + vid)

    def _vid(self, session_id: int):
        if session_id in self._vid_of:
            return self._vid_of[session_id]
        offset = self.session.id_offset
        return session_id - offset if offset <= session_id < offset + _coupling().ID_COUNT else None

    def _send(self) -> None:
        from openalea import fspm_tf as codec
        ds = self.data_structure
        inputs = [name for name, (role, _) in self._variables().items() if role == "input"]
        frame = to_frame({self.data_structure_id: ds}, geometry=False, variables=inputs)
        tables = {entity_set.name: entity_set for entity_set in frame.data_structures[0].entity_sets}
        declared = self._declared
        set_of, complex_of = {}, {}
        for name, table in tables.items():
            for vid, complex_ in zip(table.columns["id"].tolist(),
                                     table.columns.get("complex", np.full(table.count, _NO_ID)).tolist()):
                set_of[vid], complex_of[vid] = name, complex_
        out, removed = [], {}
        for name, table in tables.items():
            if name not in declared:
                continue
            vids = table.columns["id"].tolist()
            ids = [self._session_id(vid) for vid in vids]
            columns = {"id": np.array(ids, dtype=np.int64)}
            if self.rights.get(name) in ("owns", "refines"):
                labels, complexes, parents, edge_types = [], [], [], []
                for row, vid in enumerate(vids):
                    session_id = ids[row]
                    if session_id in self._echo:          # a remote entity: its topology as received
                        label, complex_, parent, edge_type = self._echo[session_id]
                    else:
                        complex_ = complex_of.get(vid, _NO_ID)
                        while complex_ != _NO_ID and set_of.get(complex_) not in declared:
                            complex_ = complex_of.get(complex_, _NO_ID)
                        complex_ = _NO_ID if complex_ == _NO_ID else self._session_id(complex_)
                        parent = int(table.columns["parent"][row]) if "parent" in table.columns else _NO_ID
                        parent = _NO_ID if parent == _NO_ID else self._session_id(parent)
                        edge_type = str(table.columns["edge_type"][row]) if "edge_type" in table.columns else "none"
                        label = table.columns["label"][row] if "label" in table.columns else name
                        label = name if label in (None, "_none") else str(label)
                    labels.append(label), complexes.append(complex_), parents.append(parent)
                    edge_types.append(edge_type)
                columns.update(label=np.array(labels, dtype=object), complex=np.array(complexes, dtype=np.int64),
                               parent=np.array(parents, dtype=np.int64),
                               edge_type=np.array(edge_types, dtype=object))
                own = {identifier for identifier in ids if identifier not in self._echo}
                gone = self._sent.get(name, set()) - own
                if gone:
                    removed[(self.data_structure_id, name)] = np.array(sorted(gone), dtype=np.int64)
                self._sent[name] = own
            for variable in inputs:
                if variable in table.columns and self._set_name(variable) == name:
                    columns[variable] = table.columns[variable]
            out.append(codec.EntitySet(name, columns=columns, rank=table.rank))
        self.session.send(codec.Frame(data_structures=[codec.DataStructure(self.data_structure_id, entity_sets=out)],
                                      update=codec.Update(self.participant, removed)))

    def _apply_next(self) -> None:
        turn = next(self._turns, None)
        if turn is None:
            self.stopped = self.session.stopped
            return
        self.turn = turn
        frame = turn.frame
        if not frame.data_structures:
            return
        structure = frame.data_structure(self.data_structure_id)
        ds = self.data_structure
        g = ds._mtg
        before = self._before_new_nodes() if self.partition_weight is not None else None
        changed = self._remove(frame) | self._create(structure, g)
        if changed:
            ds.update_topology()
        for name, (role, _) in self._variables().items():
            if role != "state_variable":
                continue
            try:
                table = structure.entity_set(self._set_name(name))
            except KeyError:
                continue
            if name not in table.columns:
                continue
            values = np.ma.filled(np.ma.asarray(table.columns[name], dtype=float), np.nan)
            vids = np.array([self._vid(int(identifier)) for identifier in table.columns["id"].tolist()],
                            dtype=object)
            known = np.array([vid is not None for vid in vids]) & ~np.isnan(values)
            if not known.any():
                continue
            location = ds.location(name)
            current = np.array(ds.get(name), dtype=float, copy=True)
            current[ds.index_of(vids[known].astype(np.int64), location=location)] = values[known]
            ds.set(name, current)
        if changed and before is not None:
            self._repartition(*before)

    def _remove(self, frame) -> bool:
        g = self.data_structure._mtg
        changed = False
        for (_, name), ids in sorted(frame.update.removed.items(),
                                     key=lambda item: -int(getattr(g.scales, item[0][1], 0))):
            for identifier in ids.tolist():
                vid = self._vid(int(identifier))
                if vid is None or vid not in g._scale:
                    continue
                g.remove_vertex(vid, reparent_child=False)
                self._vid_of.pop(int(identifier), None), self._session_of.pop(vid, None)
                self._echo.pop(int(identifier), None)
                for sent in self._sent.values():
                    sent.discard(int(identifier))
                changed = True
        return changed

    def _create(self, structure, g) -> bool:
        """The vertices of the remote entities new in the scales metafspm follows."""
        ds = self.data_structure
        changed = False
        for table in sorted(structure.entity_sets, key=lambda item: item.rank or 0):
            if self.rights.get(table.name, "follows") != "follows" or "complex" not in table.columns:
                continue
            scale = getattr(g.scales, table.name, None)
            if scale is None:
                raise ValueError(f"{type(self).__name__}: the MPG has no scale '{table.name}'")
            rows = [row for row, identifier in enumerate(table.columns["id"].tolist())
                    if self._vid(int(identifier)) is None]
            while rows:                                  # parents first
                pending = []
                for row in rows:
                    identifier = int(table.columns["id"][row])
                    parent = int(table.columns["parent"][row]) if "parent" in table.columns else _NO_ID
                    if parent != _NO_ID and self._vid(parent) is None and parent in table.columns["id"]:
                        pending.append(row)
                        continue
                    complex_ = int(table.columns["complex"][row])
                    complex_vid = self._vid(complex_) if complex_ != _NO_ID else None
                    if complex_vid is None:
                        raise ValueError(f"{type(self).__name__}: remote entity {identifier} of {table.name} has no "
                                         "complex in metafspm's MPG (follow its complex's scale)")
                    edge_type = str(table.columns["edge_type"][row]) if "edge_type" in table.columns else "none"
                    label = str(table.columns["label"][row]) if "label" in table.columns else table.name
                    properties = dict(scale=scale)
                    try:
                        properties["label"] = ds.label_code(label)
                    except ValueError:
                        pass
                    parent_vid = self._vid(parent) if parent != _NO_ID else None
                    if parent_vid is not None and g.complex(parent_vid) == complex_vid:
                        vid = g.add_child(parent_vid, **PropsConfig(edge_type=_EDGE_SYMBOLS.get(edge_type, "<"),
                                                                    **properties))
                    elif parent_vid is not None:
                        vid = g.add_component_with_topo(complex_vid, parent_vid, **PropsConfig(
                            edge_type=_EDGE_SYMBOLS.get(edge_type, "+"), **properties))
                    else:
                        vid = g.add_component(complex_vid, **PropsConfig(edge_type="/", **properties))
                    self._vid_of[identifier], self._session_of[vid] = vid, identifier
                    self._echo[identifier] = (label, complex_, parent, edge_type)
                    changed = True
                if len(pending) == len(rows):
                    raise ValueError(f"{type(self).__name__}: remote entities of {table.name} whose parents never "
                                     "come")
                rows = pending
        return changed

    def triangles(self, where=None):
        """No geometry of its own: the remote model's is not drawn by metafspm's light models."""
        from openalea.metafspm.coupling.geometry import Triangles
        return Triangles(np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int64))
