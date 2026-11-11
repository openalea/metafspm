"""
metafspm in an encoding3d coupling session (openalea.encoding3d.coupling, specification §11): another platform's model
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
are sent to the session; "follows" (and "refines", where another participant refines too), the entities the other
participants create are created in the MPG (under their complex, after their parent), and removed when they are. When remote entities appear, metafspm shares its own
variables over them by its repartition rules (StructuralComponent's), with *partition_weight* (a variable the
remote model provides, e.g. a length): an extensive quantity split, a concentration diluted.

Ids: the session id of a vertex metafspm created is its vid plus the participant's offset; the vertices created for
remote entities keep the remote ids in a map.

When metafspm's model works on a plant another platform builds (metafspm follows its scales, owning none), there is no
plant to start from: `join()` joins the session first and builds the DataStructure from the first turn's entities,
then the components are made on it and `start()` carries on::

    ds = RemoteRoots.join(endpoint)                    # metafspm's MPG and DataStructure, from the session's plant
    uptake, remote = WaterUptake(ds), RemoteRoots(ds)
    remote.start()
"""
from dataclasses import dataclass, fields

import numpy as np

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.io.encoding3d import _KINDS, _location_scales, _scale_names, to_frame, ucum_unit

_EDGE_SYMBOLS = {"successor": "<", "branch": "+"}
_NO_ID = -1
_JOINED = {}                    # id(DataStructure): the session join() opened, which start() carries on


def _coupling():
    try:
        from openalea.encoding3d import coupling
    except ImportError:
        raise ImportError("openalea.encoding3d (with its coupling) is not installed: pip install -e ../encoding3d") from None
    return coupling


@dataclass
class RemoteComponent(StructuralComponent):
    """Another platform's model in a coupling session (see the module's documentation)."""

    participant = "metafspm"            # this metafspm model's name in the session
    data_structure_id = "plant"         # the session's data structure this DataStructure is
    platform = "metafspm"
    rights = {}                         # {scale name: "owns" | "refines" | "follows"}
    links = ()                          # the links proposed (encoding3d.coupling.Link)
    endpoint = None                     # the session (default: the environment's ENCODING3D_ENDPOINT)
    timeout = 600.                      # seconds waited for a turn

    # ── what it declares ───────────────────────────────────────────────────────────
    def _variables(self) -> dict:
        """{name: (role, metadata)} of its state variables and inputs."""
        return {item.name: (item.metadata.get("role"), dict(item.metadata)) for item in fields(type(self))
                if item.metadata.get("role") in ("state_variable", "input_variable")}

    def _set_name(self, name: str) -> str:
        """The entity set (scale name) of variable *name*'s location."""
        ds = self.data_structure
        return _scale_names(ds.structure)[_location_scales(ds)[ds.location(name)]]

    @classmethod
    def _declared_variables(cls) -> dict:
        return {item.name: (item.metadata.get("role"), dict(item.metadata)) for item in fields(cls)
                if item.metadata.get("role") in ("state_variable", "input_variable")}

    @classmethod
    def _declaration(cls, set_name):
        coupling = _coupling()
        from openalea.encoding3d import vocabulary
        standard_names = {key: value for key, value in vocabulary.platform_names("metafspm").items() if value}
        declaration = coupling.Declaration(cls.participant, cls.platform)
        for name, rights in cls.rights.items():
            declaration.entity_sets.append(coupling.EntitySetSpec(cls.data_structure_id, name, rights=rights))
        for name, (role, meta) in cls._declared_variables().items():
            declaration.variables.append(coupling.VariableSpec(
                cls.data_structure_id, set_name(name, meta), name,
                "requires" if role == "state_variable" else "provides", ucum_unit(meta.get("unit")),
                standard_names.get(name), _KINDS.get(meta.get("kind")), optional=False))
        return declaration.complete()

    def declaration(self):
        return self._declaration(lambda name, meta: self._set_name(name))

    @classmethod
    def join(cls, endpoint: str = None, from_scale: str = "SubOrgan"):
        """
        Join the session before any DataStructure exists, when metafspm works on a plant another platform builds:
        the first turn's entities (of the scales the class declares) become an MPG, under the MPG's anchors where
        they have no complex, and its MPGDataStructure (*from_scale*: its nodes' scale) is returned. A RemoteComponent
        made on it then carries on the session with start(). The variables' scales come from their declarations
        (scale=).
        """
        from openalea.metafspm.coupling.declaration import scale_names
        from openalea.metafspm.data_structure.data_api import MPGDataStructure
        from openalea.metafspm.data_structure.mpg import MPG
        coupling = _coupling()
        names = scale_names()

        def set_name(name, meta):
            if meta.get("scale") is None:
                raise ValueError(f"{cls.__name__}.{name}: join() needs each variable's scale (scale=)")
            return names[meta["scale"]]

        declaration = cls._declaration(set_name)
        session = coupling.Participant.connect(declaration, endpoint or cls.endpoint, links=cls.links,
                                               timeout=cls.timeout)
        turns = session.turns()
        turn = next(turns, None)
        if turn is None:
            raise RuntimeError(f"{cls.__name__}: the session stopped before its first turn ({session.stopped})")
        g = MPG()
        maps = dict(vid_of={}, session_of={}, echo={})
        if turn.frame.data_structures:
            _create_entities(g, turn.frame.data_structure(cls.data_structure_id), cls.rights, maps,
                             session.id_offset)
        scale = getattr(g.scales, from_scale)
        if not any(g._scale.get(vid) == scale for vid in maps["session_of"]):
            raise ValueError(f"{cls.__name__}: the session's first turn holds no {from_scale} entity to build the "
                             "DataStructure on")
        g.populate_graph(scale)
        g.convert_properties_to_arraydict()
        ds = MPGDataStructure(g, from_scale=scale)
        _JOINED[id(ds)] = dict(session=session, turns=turns, turn=turn,
                               declared={entity_set.name for entity_set in declaration.entity_sets}, **maps)
        return ds

    # ── the session ────────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Join the session: the initial turn's entities and inputs applied, metafspm's initial state sent, the first
        step's inputs applied."""
        coupling = _coupling()
        self._sent = {}                           # entity set: the session ids of metafspm's entities sent last
        self._vid_of, self._session_of = {}, {}   # remote session id <-> vid of the vertex created for it
        self._echo = {}                           # remote session id: the topology it was received with
        self.stopped = None
        joined = _JOINED.pop(id(self.data_structure), None)
        if joined is not None:                    # join() opened the session and built the DataStructure
            self.session, self._turns, self._declared = joined["session"], joined["turns"], joined["declared"]
            self._vid_of, self._session_of, self._echo = joined["vid_of"], joined["session_of"], joined["echo"]
            self._apply(joined["turn"])
            self._send()
            self._apply_next()
            return
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
        from openalea import encoding3d as codec
        ds = self.data_structure
        inputs = [name for name, (role, _) in self._variables().items() if role == "input_variable"]
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
        self._apply(turn)

    def _apply(self, turn) -> None:
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
        """The vertices of the remote entities new in the scales metafspm follows or refines (another participant may
        refine them too)."""
        maps = dict(vid_of=self._vid_of, session_of=self._session_of, echo=self._echo)
        return _create_entities(g, structure, self.rights, maps, self.session.id_offset)

    def triangles(self, where=None):
        """No geometry of its own: the remote model's is not drawn by metafspm's light models."""
        from openalea.metafspm.coupling.geometry import Triangles
        return Triangles(np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int64))


def _create_entities(g, structure, rights: dict, maps: dict, id_offset: int) -> bool:
    """The MPG vertices of the remote entities of *structure* new to *maps* (vid_of, session_of, echo), in the scales
    *rights* does not own: under their complex's vertex (the MPG's anchor of their scale when they have none, and
    vertices of metafspm's own for the scales skipped in between), after their parent; their labels by the MPG's label
    codes where it knows them."""
    vid_of, session_of, echo = maps["vid_of"], maps["session_of"], maps["echo"]
    count = _coupling().ID_COUNT

    def vid(session_id):
        if session_id in vid_of:
            return vid_of[session_id]
        return session_id - id_offset if id_offset <= session_id < id_offset + count else None

    changed = False
    for table in sorted(structure.entity_sets, key=lambda item: item.rank or 0):
        if rights.get(table.name, "follows") == "owns":
            continue
        scale = getattr(g.scales, table.name, None)
        if scale is None:
            raise ValueError(f"the MPG has no scale '{table.name}'")
        ids = table.columns["id"].tolist()
        rows = [row for row, identifier in enumerate(ids) if vid(int(identifier)) is None]
        while rows:                                      # parents first
            pending = []
            for row in rows:
                identifier = int(ids[row])
                parent = int(table.columns["parent"][row]) if "parent" in table.columns else _NO_ID
                if parent != _NO_ID and vid(parent) is None and parent in ids:
                    pending.append(row)
                    continue
                complex_ = int(table.columns["complex"][row]) if "complex" in table.columns else _NO_ID
                complex_vid = vid(complex_) if complex_ != _NO_ID else None
                if complex_ != _NO_ID and complex_vid is None:
                    raise ValueError(f"remote entity {identifier} of {table.name}: its complex {complex_} is not in "
                                     "metafspm's MPG (follow its complex's scale)")
                edge_type = str(table.columns["edge_type"][row]) if "edge_type" in table.columns else "none"
                label = str(table.columns["label"][row]) if "label" in table.columns else table.name
                properties = dict(scale=scale)            # the MPG's scale property, which its index reads
                group = getattr(g.labels, table.name, None)       # the label's code: by its value, or its name
                code = g.labels.filters.get(label, getattr(group, label, None) if group is not None else None)
                if isinstance(code, (int, np.integer)):
                    properties["label"] = int(code)
                parent_vid = vid(parent) if parent != _NO_ID else None
                # its container one scale up: its complex's vertex (the MPG's anchor of its scale when it has none),
                # through vertices of metafspm's own for the scales between (another platform may skip some: an
                # organ right under its plant), which the session never sees
                container = complex_vid if complex_vid is not None else g.scales.anchors[scale]
                while g.scale(container) < scale - 1:
                    container = g.add_components_bulk(container, 1, scale=g.scale(container) + 1, edge_type="/",
                                                      isanchor=False)[0]
                edge = _EDGE_SYMBOLS.get(edge_type, "<") if parent_vid is not None else "/"
                new = g.add_components_bulk(container, 1, topo_parents=[parent_vid], edge_type=edge, isanchor=False,
                                            **properties)[0]
                vid_of[identifier], session_of[new] = new, identifier
                echo[identifier] = (label, complex_, parent, edge_type)
                changed = True
            if len(pending) == len(rows):
                raise ValueError(f"remote entities of {table.name} whose parents never come")
            rows = pending
    return changed
