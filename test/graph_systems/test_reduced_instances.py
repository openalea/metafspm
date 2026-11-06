"""
A reduction's groups drawn as instances of their templates (io.encoding3d's display="instances", the default): not
anatomy-specific. A toy plant whose segments each hold a small network (no Layer, no Cell: two pipe nodes joined
across segments, an interior ladder, a soil node and two probe nodes merged into one) is reduced, its kinds by the
segment's length in classes (Relative: a segment between two classes the blend of their kinds). A display component
draws each node as a triangle: display_triangles in the scene's coordinates, display_instances once per template in
the template's own coordinates with a 4×4 per segment. The frame writes one instance set: the template (a shared
part), each kind's recovery operator (a shared part, dense or its sparse system), per segment its transform and its
kept values; encoding3d.evaluate rebuilds the unknown on every node of the segments asked, equal to
Reduction.recover_interiors' (the kept nodes their own values, a merged member its node's).
"""
from dataclasses import dataclass
import os

import numpy as np
import pytest
from scipy.sparse import diags

from openalea import encoding3d
from openalea.encoding3d.validate import validate_bytes, validate_file

from openalea.metafspm.coupling.component import FunctionalComponent, StructuralComponent, parameter, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.geometry import TemplateShapes, Triangles  # noqa: F401
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.io.encoding3d import DISPLAY_ONLY, default_views, filter_frame, from_frame, to_frame
from openalea.metafspm.scene.population import Planting, planting, population_data_structure
from openalea.metafspm.solve import reduction as reduction_module
from openalea.metafspm.solve.decorator import boundary_set, graph_jacobian, graph_system, node_balance
from openalea.metafspm.solve.reduction import Reduction, Relative

from growth import DOC

N_SEGMENTS = 5
PIPE, LADDER, SOIL, PROBE = 0., 1., 2., 3.
# the network of a segment: its nodes (name, kind, port, merge group) and edges
NODES = (("p0", PIPE, 0., np.nan), ("p1", PIPE, 1., np.nan), ("a0", LADDER, np.nan, np.nan),
         ("a1", LADDER, np.nan, np.nan), ("a2", LADDER, np.nan, np.nan), ("a3", LADDER, np.nan, np.nan),
         ("s", SOIL, np.nan, np.nan), ("m0", PROBE, np.nan, 1.), ("m1", PROBE, np.nan, 1.))
EDGES = (("p0", "a0"), ("a0", "a1"), ("a1", "p1"), ("p0", "a2"), ("a2", "a3"), ("a3", "p1"), ("a1", "s"),
         ("a3", "s"), ("a0", "m0"), ("a2", "m1"), ("m0", "a1"), ("m1", "a3"))
# each node's place in the segment's own plane (template coordinates)
PLACE = {"p0": (0., 0.), "p1": (3., 0.), "a0": (1., 1.), "a1": (2., 1.), "a2": (1., -1.), "a3": (2., -1.),
         "s": (2.5, 2.), "m0": (1.5, 1.5), "m1": (1.5, -1.5)}
SYSTEM = "NetworkHydraulics._hydraulics"
CLASSES = Relative(0.1)


def _network(g, segment) -> dict:
    """The nodes and edges of *segment*'s network, anchored to it (no Layer, no Cell); {name: vid}."""
    s, L = g.scales, g.labels
    node_anchor, edge_anchor = s.anchors[s.Compartment], s.anchors[s.Connection]
    vids = {}
    for name, kind, port, group in NODES:
        vids[name] = g.add_component_with_topo(node_anchor, segment, **PropsConfig(
            scale=s.Compartment, edge_type='/', label=L.Compartment.Symplastic if kind == PIPE
            else L.Compartment.Apoplastic, kind=kind, port=port, merge_group=group))
    for a, b in EDGES:
        g.add_component_with_topo(edge_anchor, segment, **PropsConfig(scale=s.Connection, edge_type='/',
                                                                      label=L.Connection.Apoplastic, n_id_a=vids[a],
                                                                      n_id_b=vids[b]))
    return vids


_TEMPLATE = {}


def network_template():
    """The SectionTemplate of a segment's network and each named node's local row."""
    if not _TEMPLATE:
        from openalea.metafspm.data_structure.instances import SectionTemplate
        from openalea.metafspm.data_structure.mpg import MPG
        scratch = MPG()
        segment = scratch._add_system_root_at_scale(scratch.scales.SubOrgan,
                                                    label=scratch.labels.SubOrgan.RootSegment)
        first = scratch._id + 1
        names = _network(scratch, segment)
        scratch.convert_properties_to_arraydict()
        template = SectionTemplate.capture(scratch, segment, range(first, scratch._id + 1), key="toy network")
        _TEMPLATE.update(template=template, rows={name: vid - first for name, vid in names.items()})
    return _TEMPLATE["template"], _TEMPLATE["rows"]


def wiring(g):
    """The pipe nodes of equal port joined from a segment to the next."""
    return [dict(node_label=g.labels.Compartment.Symplastic, edge_label=g.labels.Connection.Apoplastic,
                 ordering="port", match="equal")]


class NetworkSegments(StructuralComponent):
    """A chain of root segments, each an instance of the toy network (reduced when the plant declares it)."""

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        s, L = g.scales, g.labels
        parent = plant
        for scale, label in ((s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root), (s.Phytomer, L.Phytomer.Root),
                             (s.Organ, L.Organ.RootInternode)):
            parent = g.add_component(parent, **PropsConfig(scale=scale, edge_type='/', label=label))
        template = network_template()[0]
        reduction = getattr(g, "reduction", None)
        previous = None
        for _ in range(N_SEGMENTS):
            props = PropsConfig(scale=s.SubOrgan, edge_type='<' if previous else '/', label=L.SubOrgan.RootSegment)
            previous = g.add_child(previous, **props) if previous else g.add_component(parent, **props)
            reduced = reduction is not None and reduction.reduces(g, previous)
            g.add_instance(previous, reduction.template_for(template) if reduced else template)


def _inlet(ds):
    """The first segment's pipe node of port 0 (held)."""
    return (ds.get("kind") == PIPE) & (ds.get("port") == 0.) & (ds.owner("SubOrgan") == 0)


def _conductances(view, K, kind, length) -> np.ndarray:
    """Each edge's conductance: K times its segment's length on the edges reaching the soil node."""
    ends = abs(view.incidence)
    on_soil = np.asarray(ends.T @ (np.asarray(kind) == SOIL).astype(float)).reshape(-1) > 0
    edge_length = 0.5 * np.asarray(ends.T @ np.asarray(length, dtype=float)).reshape(-1)
    return np.asarray(K, dtype=float) * np.where(on_soil, edge_length, 1.)


@dataclass
class NetworkHydraulics(FunctionalComponent):
    """Steady water potential on the network: the soil nodes joined to the soil, the inlet held."""
    water_potential: float = state_variable(**DOC, default=-0.5, location="node")
    K: float = parameter(**DOC, by="", default=1., location="edge")
    kind: float = parameter(**DOC, by="", default=0., location="node")
    port: float = parameter(**DOC, by="", default=np.nan, location="node")
    length: float = parameter(**DOC, by="", default=1., location="node")
    soil_water_potential: float = parameter(**DOC, by="", default=-0.2, location="node")
    inlet_water_potential: float = parameter(**DOC, by="", default=-1., location="node")

    @graph_system(node_unknowns=["water_potential"], solver="newton")
    class _hydraulics:
        soil = boundary_set(filters={"kind": "==2"}, kind="robin", value="soil_water_potential", weight=0.6)
        inlet = boundary_set(filters=_inlet, kind="dirichlet", value="inlet_water_potential")

        @node_balance(field="water_potential")
        def _balance(self, water_potential, K, kind, length):
            B = self.graph_view.incidence
            conductance = _conductances(self.graph_view, K, kind, length)
            return np.asarray(B @ (conductance * np.asarray(B.T @ water_potential).reshape(-1))).reshape(-1)

        @graph_jacobian
        def _jacobian(self, K, kind, length):
            B = self.graph_view.incidence
            return (B @ diags(_conductances(self.graph_view, K, kind, length)) @ B.T).tocsr()


def _of_kind(value):
    def rule(template):
        rows, values = template.properties["kind"]
        out = np.zeros(template.size, dtype=bool)
        out[rows] = np.asarray(values, dtype=float) == value
        return out
    rule.__name__ = f"kind_{int(value)}"
    return rule


_pipes, _soil, _probes = _of_kind(PIPE), _of_kind(SOIL), _of_kind(PROBE)


def _model(reduced=True, merge=True):
    class NetworkPlant(CompositeModel):
        from_scale, nodes, initiators = "SubOrgan", "Compartment", (NetworkSegments,)
        reduction = Reduction(location="SubOrgan", where={"label": "RootSegment"}, ports=_pipes,
                              keep=(_soil, _probes), kind={"length": CLASSES}, systems=(SYSTEM,),
                              merge={"where": _probes, "by": ("merge_group",)} if merge else None) \
            if reduced else None

        @staticmethod
        def wiring(g):
            return wiring(g)

        def __init__(self, data_structure, time_step):
            self.hydraulics = NetworkHydraulics(data_structure=data_structure, time_step=time_step)
            self.couple(self.hydraulics)

        def run(self):
            self.hydraulics()
    return NetworkPlant


LENGTHS = (1., 1.1, 1.05, 1.2, 0.97)          # class values and lengths between two classes (blends)


def _solved(reduced=True, merge=True):
    model = _model(reduced, merge)
    population = population_data_structure(planting([Planting()], model, {}), model)
    ds = population.data_structure
    ds.set("length", np.asarray(LENGTHS, dtype=float)[np.asarray(ds.owner("SubOrgan"))])
    population.instance.run()
    return population


@dataclass
class NetworkDrawer(FunctionalComponent):
    """Draws each node of a segment as a small triangle at its place in the segment's plane, the segment k-th along
    z (its plane scaled by its length): display_triangles in the scene's coordinates (a reduced segment whole, its
    interior by display id), display_instances once per template. No variable of its own."""

    @property
    def ds(self):
        return self.data_structure

    def _transform(self, segment) -> np.ndarray:
        ds = self.ds
        position = int(np.searchsorted(ds.entity_ids("SubOrgan"), segment))
        transform = np.diag([LENGTHS[position], LENGTHS[position], 1., 1.])
        transform[:3, 3] = (0.5, -0.25, float(position))
        return transform

    @staticmethod
    def _shape(rows) -> Triangles:
        """A triangle per template row of *rows* (Compartments), at its node's place."""
        names = {row: name for name, row in network_template()[1].items()}
        corners = np.array([[[0., 0., 0.], [0.2, 0., 0.], [0., 0.2, 0.]]])
        vertices = np.concatenate([corners + np.array([*PLACE[names[row]], 0.]) for row in rows])
        return Triangles(vertices, np.asarray(rows, dtype=np.int64))

    def display_instances(self, owners):
        ds = self.ds
        reduction = ds.structure.reduction
        reduced = {int(owner): key for owner, key in reduction.reduced_groups(ds).values()}
        drawn = [int(owner) for owner in owners if int(owner) in reduced]
        shapes = {}
        for owner in drawn:
            full = reduction.full_template(reduced[owner])
            shapes.setdefault(full.key, self._shape(np.flatnonzero(full.scale == scales.Compartment)))
        return TemplateShapes(drawn, shapes, [reduction.full_template(reduced[owner]).key for owner in drawn],
                              np.array([self._transform(owner) for owner in drawn]).reshape(-1, 4, 4))

    def display_triangles(self, where=None):
        from openalea.metafspm.solve.reduction import display_ids
        ds = self.ds
        g = ds.structure
        reduction = g.reduction
        wanted = np.ones(ds.n_nodes(), dtype=bool) if where is None else np.asarray(where, dtype=bool)
        segments = ds.entity_ids("SubOrgan")
        owner = np.asarray(ds.owner("SubOrgan"))
        parts = []
        reduced = {int(o): key for o, key in reduction.reduced_groups(ds).values()} if reduction is not None else {}
        template, _ = network_template()
        compartments = np.flatnonzero(template.scale == scales.Compartment)
        for position, segment in enumerate(segments.tolist()):
            nodes = np.flatnonzero((owner == position) & wanted)
            if segment in reduced:
                if not nodes.size:
                    continue
                held = reduction.full_rows(reduced[segment])
                interior = np.setdiff1d(compartments, held)
                rows = np.concatenate([held[np.isin(held, compartments)], interior])
                shape = self._shape(rows)
                ids = np.concatenate([np.flatnonzero(owner == position), np.full(interior.size, -1)])
                owner_ids = np.concatenate([ds.entity_ids("node")[np.flatnonzero(owner == position)],
                                            display_ids(segment, interior)])
            else:
                rows, ids = compartments, nodes
                shape = self._shape(rows)
                owner_ids = ds.entity_ids("node")[nodes]
            transform = self._transform(segment)
            points = shape.vertices.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]
            parts.append(Triangles(points.reshape(-1, 3, 3), ids, owner_ids))
        return Triangles.concatenate(parts) if parts else Triangles.empty()


@pytest.fixture(scope="module")
def solved():
    population = _solved()
    return population.data_structure, population.instance


def _frame(ds, instance, **options):
    return to_frame({"plants": ds}, components={"plants": list(instance.components) + [NetworkDrawer(data_structure=ds)]},
                    **options)


def _expected(ds):
    """{segment vid: the unknown on its full template's Compartments}: recover_interiors' on the interior (a merged
    member its node's value), the kept nodes' own."""
    reduction = ds.structure.reduction
    out = {}
    values = np.asarray(ds.get("water_potential"))
    owner = np.asarray(ds.owner("SubOrgan"))
    for i, (segment, key, full, interior_rows, interior) in reduction.recover_interiors(ds).items():
        rows = np.flatnonzero(full.scale == scales.Compartment)
        field = np.full(rows.size, np.nan)
        held = reduction.full_rows(key)
        kept = held[np.isin(held, rows)]
        field[np.searchsorted(rows, kept)] = values[owner == i]
        field[np.searchsorted(rows, interior_rows)] = interior
        out[segment] = field
    return out


def test_the_groups_are_instances_whose_derived_field_is_the_recovery(solved):
    ds, instance = solved
    frame = _frame(ds, instance)
    plants = frame.data_structures[0]
    assert [instance_set.name for instance_set in plants.instance_sets] == ["NetworkDrawer"]
    sections = plants.instance_set("NetworkDrawer")
    assert sections.owners == "SubOrgan"
    np.testing.assert_array_equal(sections.columns["owner"], ds.entity_ids("SubOrgan"))
    assert DISPLAY_ONLY not in plants.entity_set("Compartment").columns       # no interior rows: the instances
    templates = [part for part in sections.parts if isinstance(part, encoding3d.Template)]
    operators = [part for part in sections.parts if isinstance(part, encoding3d.Operator)]
    assert len(templates) == 1                                       # one template, shared by every segment
    field = sections.derived_field("water_potential")
    assert field.entity_set == "Compartment" and field.input == "water_potential_kept"
    assert field.operators.shape[1] == 2                             # a length between classes: two kinds
    between = (field.operators >= 0).sum(axis=1) == 2
    assert between.any() and (~between).any()
    np.testing.assert_allclose(field.weights.sum(axis=1), 1.)
    assert len(operators) == len(np.unique(field.operators[field.operators >= 0]))   # each kind once
    expected = _expected(ds)
    evaluated = encoding3d.evaluate(frame, "water_potential")
    for owner, values in zip(sections.columns["owner"].tolist(), evaluated):
        np.testing.assert_allclose(values, expected[owner], rtol=1e-12, atol=1e-14)
    # the template's nodes: their rows' constant properties (label names, kind, port)
    nodes = templates[0].entity_set("Compartment").columns
    template, rows = network_template()
    np.testing.assert_array_equal(nodes["id"], np.flatnonzero(template.scale == scales.Compartment))
    assert set(nodes["label"].tolist()) == {"Symplastic", "Apoplastic"}
    np.testing.assert_array_equal(nodes["kind"], [kind for _, kind, _, _ in NODES])
    assert validate_bytes(encoding3d.to_bytes(frame, embed_parts=True)).ok


def test_a_merged_member_has_its_node_s_value_and_the_kept_nodes_their_own(solved):
    ds, instance = solved
    frame = _frame(ds, instance)
    _, rows = network_template()
    compartments = np.flatnonzero(network_template()[0].scale == scales.Compartment)
    m0, m1 = (int(np.searchsorted(compartments, rows[name])) for name in ("m0", "m1"))
    values = np.asarray(ds.get("water_potential"))
    owner = np.asarray(ds.owner("SubOrgan"))
    for position, field in enumerate(encoding3d.evaluate(frame, "water_potential")):
        assert field[m0] == field[m1]                                   # m1 merged into m0
        assert np.isin(values[owner == position], field).all()          # the kept nodes' own values


def test_placed_instances_draw_what_display_triangles_draws(solved):
    ds, instance = solved
    frame = _frame(ds, instance)
    placed = encoding3d.placed_meshes(frame, "plants", "NetworkDrawer")
    whole = NetworkDrawer(data_structure=ds).display_triangles()
    drawn = NetworkDrawer(data_structure=ds).display_instances(ds.entity_ids("SubOrgan")).placed()
    np.testing.assert_allclose(np.sort(drawn.vertices.reshape(-1)), np.sort(whole.vertices.reshape(-1)))
    corners = placed.positions[placed.triangles]                       # the frame's, placed by its transforms
    assert corners.shape == whole.vertices.shape
    np.testing.assert_allclose(np.sort(corners.reshape(-1)), np.sort(whole.vertices.reshape(-1)))


def test_full_and_none_displays(solved):
    ds, instance = solved
    full = _frame(ds, instance, display="full").data_structures[0]
    assert not full.instance_sets and np.asarray(full.entity_set("Compartment").columns[DISPLAY_ONLY]).any()
    none = _frame(ds, instance, display=None).data_structures[0]
    assert not none.instance_sets and DISPLAY_ONLY not in none.entity_set("Compartment").columns
    assert "NetworkDrawer" not in {mesh.name for mesh in none.meshes}     # every segment reduced: none drawn
    instances = _frame(ds, instance).data_structures[0]
    assert "NetworkDrawer" not in {mesh.name for mesh in instances.meshes}  # the reduced groups not in the mesh
    with pytest.raises(ValueError, match="display"):
        _frame(ds, instance, display="whole")


def test_an_instances_frame_is_rebuilt_and_filtered(solved):
    ds, instance = solved
    rebuilt = from_frame(_frame(ds, instance))["plants"]
    np.testing.assert_array_equal(rebuilt.entity_ids("node"), ds.entity_ids("node"))
    np.testing.assert_allclose(rebuilt.get("water_potential"), ds.get("water_potential"))
    segment = int(ds.entity_ids("SubOrgan")[2])
    kept = filter_frame(_frame(ds, instance), entities=[segment]).data_structures[0].instance_set("NetworkDrawer")
    assert kept.columns["owner"].tolist() == [segment]
    assert kept.derived_field("water_potential").operators.shape[0] == 1 and len(kept.vectors["water_potential_kept"]) == 1
    frame = _frame(ds, instance)
    expected = encoding3d.evaluate(frame, "water_potential")[2]
    filtered = filter_frame(_frame(ds, instance), entities=[segment])
    np.testing.assert_allclose(encoding3d.evaluate(filtered, "water_potential")[0], expected)
    dropped = filter_frame(_frame(ds, instance), scales=["Plant", "Axis"]).data_structures[0]
    assert not dropped.instance_sets                                   # their owners' scale dropped


def test_the_sparse_system_gives_the_dense_operator(solved, monkeypatch):
    """Above DISPLAY_DENSE_MAX a kind keeps no dense recovery: its operator is its sparse system, solved by the
    reader, the same field."""
    ds, instance = solved
    dense = encoding3d.evaluate(_frame(ds, instance), "water_potential")
    monkeypatch.setattr(reduction_module, "DISPLAY_DENSE_MAX", 0)
    population = _solved()
    frame = _frame(population.data_structure, population.instance)
    operators = [part for part in frame.data_structures[0].instance_set("NetworkDrawer").parts
                 if isinstance(part, encoding3d.Operator)]
    assert {operator.format for operator in operators} == {"solve"}
    for left, right in zip(encoding3d.evaluate(frame, "water_potential"), dense):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-14)


def test_parts_are_written_once_and_the_views_declare_the_instances(solved, tmp_path):
    ds, instance = solved
    first, second = str(tmp_path / "frame_000000.glb"), str(tmp_path / "frame_000001.glb")
    keys = encoding3d.write(first, _frame(ds, instance, time_unit="s"))
    parts = sorted(os.listdir(tmp_path / "parts"))
    assert parts == sorted(f"{key}.glb" for key in keys)
    times = {name: os.path.getmtime(tmp_path / "parts" / name) for name in parts}
    assert encoding3d.write(second, _frame(ds, instance, time=1., step=1, time_unit="s")) == keys   # not again
    assert {name: os.path.getmtime(tmp_path / "parts" / name) for name in parts} == times
    encoding3d.write_index(str(tmp_path / "scene.fspm.json"), [{"time": 0., "step": 0, "file": "frame_000000.glb"},
                                                               {"time": 1., "step": 1, "file": "frame_000001.glb"}],
                           time_unit="s", parts=keys)
    report = validate_file(str(tmp_path / "scene.fspm.json"))
    assert report.ok, [issue for issue in report.issues if issue.severity == "error"]
    read = encoding3d.read(first)
    for left, right in zip(encoding3d.evaluate(read, "water_potential"), _expected(ds).values()):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-14)


def test_default_views_declare_the_instanced_layer(solved):
    from openalea.encoding3d import views
    ds, instance = solved
    frame = _frame(ds, instance)
    layers = {}
    from openalea.metafspm.io.encoding3d import _plant_layers, _declarations
    for layer in _plant_layers(frame.data_structures[0], ds, _declarations(instance.components)):
        layers[layer["name"]] = layer
    layer = layers["plants NetworkDrawer instances"]
    assert layer["source"] == {"data_structure": "plants", "kind": "instances", "instance_set": "NetworkDrawer"}
    assert layer["color_by"] == {"entity_set": "Compartment", "variable": "water_potential"}
    document = {"encoding3d_views": 1, "active": "default", "configurations": {"default": {"layers": [layer]}}}
    assert views.check_views(document, frame).ok


def test_without_a_reduction_nothing_is_instanced():
    population = _solved(reduced=False)
    ds = population.data_structure
    plants = to_frame({"plants": ds}, components={"plants": list(population.instance.components)
                                                  + [NetworkDrawer(data_structure=ds)]}).data_structures[0]
    assert not plants.instance_sets
    assert "NetworkDrawer" in {mesh.name for mesh in plants.meshes}


def _recorded_scene(out, **options):
    from openalea.metafspm.scene.scene import Scene, SceneRecorder
    model = _model()
    scene = Scene(planting=planting([Planting()], model, {}), time_step=3600.,
                  recorder=SceneRecorder(out, summaries=False, frame_period=1, display_period=1, **options))
    population = scene.populations[0]
    population.instance.components.append(NetworkDrawer(data_structure=population.data_structure))
    return scene


def test_the_recorder_lists_the_parts_prunes_them_on_resume_and_packs_the_frames(tmp_path):
    import json
    import types
    out = str(tmp_path / "run")
    scene = _recorded_scene(out, pack_at_finish=True)
    for _ in range(2):
        scene.run()
    scene.finish()
    frames = os.path.join(out, "frames")
    index = encoding3d.read_index(os.path.join(frames, "scene.fspm.json"))
    used = set()
    for entry in index["frames"]:
        used |= set(encoding3d.codec.frame_part_keys(entry["path"]))
        assert encoding3d.read(entry["path"]).data_structure(scene.populations[0].name).instance_sets
    assert used and set(index["parts"]) == used
    assert sorted(os.listdir(os.path.join(frames, "parts"))) == sorted(f"{key}.glb" for key in used)
    assert validate_file(os.path.join(frames, "scene.fspm.json")).ok
    with open(os.path.join(out, "scenario.json")) as file:
        recorder = json.load(file)["recorder"]
    assert (recorder["display"], recorder["pack_at_finish"], recorder["pack_compression"]) == ("instances", True,
                                                                                               "deflate")
    archive = os.path.join(out, "frames.fspm.zip")                  # read as the folder
    last = encoding3d.read_index(archive)["frames"][-1]["path"]
    np.testing.assert_allclose(np.concatenate(encoding3d.evaluate(encoding3d.read(last), "water_potential")),
                               np.concatenate(encoding3d.evaluate(encoding3d.read(index["frames"][-1]["path"]),
                                                                  "water_potential")))
    with open(os.path.join(out, "run.json")) as file:
        assert json.load(file)["frames_archive"] == "frames.fspm.zip"
    stray = os.path.join(frames, "parts", "0" * 64 + ".glb")         # a part no frame kept refers to
    with open(stray, "wb") as file:
        file.write(b"")
    scene.recorder.resume_at(types.SimpleNamespace(iteration=1, time=3600.))
    resumed = encoding3d.read_index(os.path.join(frames, "scene.fspm.json"))
    assert [entry["step"] for entry in resumed["frames"]] == [0, 1] and not os.path.exists(stray)
    assert set(resumed["parts"]) == used


def test_the_recorder_s_display_options_are_checked(tmp_path):
    from openalea.metafspm.scene.scene import SceneRecorder
    with pytest.raises(ValueError, match="display"):
        SceneRecorder(str(tmp_path), frame_period=1, display="whole")
    with pytest.raises(ValueError, match="frame_period"):
        SceneRecorder(str(tmp_path), pack_at_finish=True)
    with pytest.raises(ValueError, match="pack_compression"):
        SceneRecorder(str(tmp_path), frame_period=1, pack_at_finish=True, pack_compression="zstd")
    with pytest.raises(ValueError, match="display="):
        SceneRecorder(str(tmp_path), frame_period=1, frame_options={"display": "full"})


def test_a_variables_filter_keeps_the_templates_columns_it_names(solved):
    ds, instance = solved
    frame = _frame(ds, instance, variables={"plants": {"Compartment": ["water_potential", "port"], "*": []}})
    sections = frame.data_structures[0].instance_set("NetworkDrawer")
    template = next(part for part in sections.parts if isinstance(part, encoding3d.Template))
    assert set(template.entity_set("Compartment").columns) == {"id", "label", "port"}
    assert len(encoding3d.evaluate(frame, "water_potential")) == sections.count
