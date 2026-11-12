"""
metafspm in an encoding3d coupling session (openalea.metafspm.io.coupling): the root growth model coupled to a carbon
model of another platform through a RemoteComponent, the session run by encoding3d's coordinator.

The remote carbon model here is a plain encoding3d participant in a thread (any platform's would answer the same
messages): it refines each root segment into two layers, and gives each layer a hexose concentration from its
segment's structural mass; the coordinator averages them back to the segments, the growth model's C_hexose_root.
metafspm owns the plants down to the segments, and follows the layers (created in its MPG).
"""
import os
import threading
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("zmq")

from openalea import encoding3d  # noqa: E402
from openalea.encoding3d.coupling import (Coordinator, Declaration, EntitySetSpec, Link, Participant,  # noqa: E402
                                       VariableSpec)
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable  # noqa: E402
from openalea.metafspm.solve.decorator import actual  # noqa: E402
from openalea.metafspm.data_structure.configs import ScalesConfig as scales  # noqa: E402
from openalea.metafspm.io.coupling import RemoteComponent  # noqa: E402

from growth import DOC, RootGrowthProbe  # noqa: E402
from test_encoding3d import _population  # noqa: E402

NO_ID = encoding3d.NO_ID
STEPS = 4
# The command running Julia with encoding3d, PlantGraphs and ZMQ (e.g. "julia --project=env"): the carbon model in
# VirtualPlantLab (vpl_carbon.jl)
JULIA = os.environ.get("ENCODING3D_JULIA")
VPL_CARBON = os.path.join(os.path.dirname(__file__), "vpl_carbon.jl")


@dataclass
class RemoteCarbon(RemoteComponent):
    """The remote carbon model as a metafspm component: C_hexose_root from it, struct_mass to it."""
    participant = "metafspm"
    rights = {"Plant": "owns", "Axis": "owns", "GrowthUnit": "owns", "Phytomer": "owns", "Organ": "owns",
              "SubOrgan": "owns", "Layer": "follows"}
    links = [Link("metafspm", "C_hexose_root", "carbon", "hexose", aggregation="mean")]
    C_hexose_root: float = state_variable(**dict(DOC, unit="mol.g-1"), default=1., scale=scales.SubOrgan,
                                          kind="massic_concentration")
    struct_mass: float = input_variable(**dict(DOC, unit="g"), default=0., scale=scales.SubOrgan,
                                        kind="extensive")


def hexose(layer_rank: int, struct_mass):
    return (layer_rank + 1.) / (1. + struct_mass)


def run_carbon(endpoint, log):
    """Two layers under each segment; layer k's hexose (k + 1) / (1 + its segment's structural mass). At step 3 it
    removes the first layer it made."""
    declaration = Declaration("carbon", "test", [EntitySetSpec("plant", "SubOrgan"),
                                                 EntitySetSpec("plant", "Layer", rights="refines")], [
        VariableSpec("plant", "Layer", "struct_mass", "requires", "g", kind="extensive"),
        VariableSpec("plant", "Layer", "hexose", "provides", "mol.g-1", kind="massic_concentration")])
    participant = Participant.connect(declaration, endpoint, timeout=120)
    layers = {}                                     # layer id: (segment, rank)
    for turn in participant.turns():
        plant = turn.frame.data_structure("plant")
        segments = plant.entity_set("SubOrgan").columns["id"].tolist()
        with_layers = {segment for segment, _ in layers.values()}
        for segment in segments:
            if segment not in with_layers:
                first = participant.id_offset + len(layers) + 1000 * turn.step
                layers[first], layers[first + 1] = (segment, 0), (segment, 1)
        masses = {}
        try:
            known = plant.entity_set("Layer")
            masses = dict(zip(known.columns["id"].tolist(), np.asarray(known.columns["struct_mass"]).tolist()))
        except KeyError:
            pass
        removed = {}
        if turn.step == 3:
            first = min(layers)
            removed = {("plant", "Layer"): [first]}
            del layers[first]
        ids = list(layers)
        log[turn.step] = dict(masses)
        columns = {"id": np.array(ids, dtype=np.int64),
                   "complex": np.array([layers[layer][0] for layer in ids], dtype=np.int64),
                   "parent": np.full(len(ids), NO_ID, dtype=np.int64),
                   "edge_type": np.array(["none"] * len(ids), dtype=object),
                   "label": np.array(["Layer"] * len(ids), dtype=object),
                   "hexose": np.array([hexose(layers[layer][1], np.nan_to_num(masses.get(layer, 0.)))
                                       for layer in ids])}
        participant.send(encoding3d.Frame(data_structures=[encoding3d.DataStructure("plant", entity_sets=[
            encoding3d.EntitySet("Layer", rank=7, columns=columns)])], update=encoding3d.Update("carbon", removed)))


def run_metafspm(endpoint_of, start_carbon, tmp_path, removal=True):
    """The growth model and the remote carbon model coupled for STEPS steps; checks what metafspm receives."""
    data_structure = _population()
    growth, remote = RootGrowthProbe(data_structure=data_structure), RemoteCarbon(data_structure=data_structure)
    coordinator = Coordinator(endpoint_of(tmp_path), steps=STEPS, interval=1., expect=["metafspm"],
                              links=[Link("carbon", "struct_mass", "metafspm", "struct_mass", aggregation="copy")],
                              record=str(tmp_path / "record"), timeout=300)
    remote.endpoint = coordinator.endpoint
    log = {}
    threads = start_carbon(coordinator, log)
    session = threading.Thread(target=coordinator.run, daemon=True)
    session.start()

    # explicit coupling: the carbon model creates a segment's layers at its turn, knowing their mass (copied from
    # the segment) from its next turn on; until then their hexose is that of a zero mass, 1.5
    def expected(masses, new):
        values = 1.5 / (1. + masses)
        values[new] = 1.5
        return values

    remote.start()
    assert np.allclose(data_structure.get("C_hexose_root"), 1.5)
    mtg = data_structure._mtg

    def layers():                                                   # the Layer vertices under segments (not anchors)
        nodes = set(data_structure.entity_ids("node").tolist())
        return [v for v in mtg.vertices(scale=mtg.scales.Layer) if mtg.complex(v) in nodes]

    assert len(layers()) == 2 * data_structure.n_nodes()                        # followed: created in metafspm's MPG
    assert {mtg.complex(v) for v in layers()} == set(data_structure.entity_ids("node").tolist())
    nodes = data_structure.n_nodes()
    for step in range(1, STEPS + 1):
        before = set(data_structure.entity_ids("node").tolist())
        growth()
        masses = np.array(data_structure.get("struct_mass"), dtype=float)
        new = np.array([vid not in before for vid in data_structure.entity_ids("node").tolist()])
        remote()
        values = np.asarray(data_structure.get("C_hexose_root"))
        if step == 3 and removal:        # the segment that lost its first layer: the mean of its other, 2 / (1 + m)
            differs = ~np.isclose(values, expected(masses, new))
            assert differs.sum() == 1 and np.isclose(values[differs], 2. / (1. + masses[differs])).all()
        elif step < STEPS:
            assert np.allclose(values, expected(masses, new)), step
    assert remote.stopped == "end of the session"
    assert data_structure.n_nodes() > nodes                                     # the roots grew new segments, sent each step
    # the layers of the segments before the last step (those of the last step's new segments come with a turn the
    # session no longer sends: they are in its record), one removed by the carbon model at step 3
    assert len(layers()) == 2 * len(before) - (1 if removal else 0)
    session.join(60)
    for thread in threads:
        thread.join(60)
    record = encoding3d.read_index(str(tmp_path / "record" / "scene.fspm.json"))
    assert [entry["step"] for entry in record["frames"]] == list(range(STEPS + 1))
    return log


ENDPOINTS = {"zmq": lambda folder: "tcp://127.0.0.1:*", "folder": lambda folder: f"files:{folder / 'session'}"}


@pytest.mark.parametrize("transport", sorted(ENDPOINTS))
def test_a_remote_model_as_a_component(transport, tmp_path):
    def start_carbon(coordinator, log):
        coordinator.expect.append("carbon")
        thread = threading.Thread(target=run_carbon, args=(coordinator.endpoint, log), daemon=True)
        thread.start()
        return [thread]

    log = run_metafspm(ENDPOINTS[transport], start_carbon, tmp_path)
    assert all(value > 0 for value in log[2].values())               # each layer's mass, copied from its segment


@pytest.mark.skipif(not JULIA, reason="ENCODING3D_JULIA (Julia with encoding3d, PlantGraphs and ZMQ) is not set")
@pytest.mark.parametrize("transport", sorted(ENDPOINTS))
def test_virtualplantlab_as_the_carbon_model(transport, tmp_path):
    """The same coupling, the carbon model in VirtualPlantLab: its layers a PlantGraphs graph grown by a rule."""
    def start_carbon(coordinator, log):
        coordinator.launch("carbon", f"{JULIA} {VPL_CARBON}")
        return []

    run_metafspm(ENDPOINTS[transport], start_carbon, tmp_path, removal=False)


@dataclass
class Pool(FunctionalComponent):
    """An amount per segment, metafspm's own extensive variable."""
    amount: float = state_variable(**dict(DOC, unit="mol"), default=1., scale=scales.SubOrgan, kind="extensive")


@dataclass
class RemoteGrower(RemoteComponent):
    """Another platform adding segments after metafspm's (both refine the organs into segments): their lengths come
    from it, and metafspm shares its amounts over the new segments by those lengths (partition_weight)."""
    participant = "metafspm"
    rights = {"Plant": "owns", "Axis": "owns", "GrowthUnit": "owns", "Phytomer": "owns", "Organ": "owns",
              "SubOrgan": "refines"}
    partition_weight = "length"
    length: float = state_variable(**dict(DOC, unit="m"), default=0., scale=scales.SubOrgan, kind="descriptor")


def run_grower(endpoint, log):
    """At each step from the first, one segment after each segment ending a chain, 0.5 m long."""
    declaration = Declaration("grower", "test", [EntitySetSpec("plant", "Organ"),
                                                 EntitySetSpec("plant", "SubOrgan", rights="refines")], [
        VariableSpec("plant", "SubOrgan", "length", "provides", "m", kind="descriptor")])
    participant = Participant.connect(declaration, endpoint, timeout=120)
    mine = {}                                       # segment id: (complex, parent)
    for turn in participant.turns():
        segments = turn.frame.data_structure("plant").entity_set("SubOrgan")
        ids = segments.columns["id"].tolist()
        if turn.step >= 1:
            parents = set(segments.columns["parent"].tolist())
            ends = [row for row, identifier in enumerate(ids) if identifier not in parents]
            for row in ends:
                new = participant.id_offset + len(mine)
                mine[new] = (int(segments.columns["complex"][row]), ids[row])
            log[turn.step] = len(ends)
        new_ids = list(mine)
        columns = {"id": np.array(new_ids, dtype=np.int64),
                   "complex": np.array([mine[segment][0] for segment in new_ids], dtype=np.int64),
                   "parent": np.array([mine[segment][1] for segment in new_ids], dtype=np.int64),
                   "edge_type": np.array(["successor"] * len(new_ids), dtype=object),
                   "label": np.array(["RootSegment"] * len(new_ids), dtype=object),
                   "length": np.full(len(new_ids), 0.5)}
        participant.send(encoding3d.Frame(data_structures=[encoding3d.DataStructure("plant", entity_sets=[
            encoding3d.EntitySet("SubOrgan", rank=6, columns=columns)])]))


def test_metafspm_shares_its_amounts_over_remote_segments(tmp_path):
    """C-5: the remote model's new segments among metafspm's nodes; metafspm splits its own extensive amount between
    each new segment and its parent by their lengths (StructuralComponent's repartition), the total kept."""
    data_structure = _population()
    pool, remote = Pool(data_structure=data_structure), RemoteGrower(data_structure=data_structure)
    coordinator = Coordinator("tcp://127.0.0.1:*", steps=3, interval=1., expect=["metafspm", "grower"], timeout=120)
    remote.endpoint = coordinator.endpoint
    log = {}
    threads = [threading.Thread(target=coordinator.run, daemon=True),
               threading.Thread(target=run_grower, args=(coordinator.endpoint, log), daemon=True)]
    for thread in threads:
        thread.start()
    remote.start()
    total = float(np.sum(data_structure.get("amount")))
    nodes = data_structure.n_nodes()
    for _ in range(3):
        before = dict(zip(data_structure.entity_ids("node").tolist(), np.asarray(data_structure.get("amount")).tolist()))
        lengths = dict(zip(data_structure.entity_ids("node").tolist(), np.asarray(data_structure.get("length")).tolist()))
        remote()
        if remote.stopped is not None:
            break
        ids = data_structure.entity_ids("node").tolist()
        amount = dict(zip(ids, np.asarray(data_structure.get("amount")).tolist()))
        assert np.sum(data_structure.get("amount")) == pytest.approx(total)          # the amount kept
        mtg = data_structure._mtg
        for vid in ids:
            if vid in before:
                continue                                              # a new segment: its parent's share
            parent = mtg.parent(vid)
            share = 0.5 / (0.5 + lengths[parent])
            assert amount[vid] == pytest.approx(share * before[parent])
            assert amount[parent] == pytest.approx((1. - share) * before[parent])
    assert data_structure.n_nodes() > nodes
    for thread in threads:
        thread.join(60)


@dataclass
class RemoteRoots(RemoteComponent):
    """Another platform's root architecture: metafspm follows its plant, organs and segments (it owns none)."""
    participant = "metafspm"
    rights = {"Plant": "follows", "Organ": "follows", "SubOrgan": "follows"}
    length: float = state_variable(**dict(DOC, unit="m"), default=0., scale=scales.SubOrgan, kind="extensive")
    radius: float = state_variable(**dict(DOC, unit="m"), default=0., scale=scales.SubOrgan, kind="intensive")
    water_uptake: float = input_variable(**dict(DOC, unit="m3/s"), default=0., scale=scales.SubOrgan,
                                         kind="extensive")


@dataclass
class WaterUptake(FunctionalComponent):
    """Uptake per segment, proportional to its surface."""
    length: float = input_variable(**dict(DOC, unit="m"), default=0., scale=scales.SubOrgan)
    radius: float = input_variable(**dict(DOC, unit="m"), default=0., scale=scales.SubOrgan)
    water_uptake: float = state_variable(**dict(DOC, unit="m3/s"), default=0., scale=scales.SubOrgan,
                                         kind="extensive")

    @actual
    def _water_uptake(self, length, radius):
        return 1e-6 * 2 * np.pi * radius * length


def run_roots(endpoint, log):
    """A plant (no complex), an organ, a chain of segments: one more each step, 1 cm long, radius 0.5 mm (in cm)."""
    declaration = Declaration("roots", "test", [EntitySetSpec("plant", name, rights="owns")
                                                for name in ("Plant", "Organ", "SubOrgan")], [
        VariableSpec("plant", "SubOrgan", "length", "provides", "cm", "length", "extensive"),
        VariableSpec("plant", "SubOrgan", "radius", "provides", "cm", "radius", "intensive"),
        VariableSpec("plant", "SubOrgan", "water_uptake", "requires", "m3/s", kind="extensive")])
    participant = Participant.connect(declaration, endpoint, timeout=120)
    plant, organ = participant.id_offset, participant.id_offset + 1
    segments = [participant.id_offset + 2, participant.id_offset + 3]
    for turn in participant.turns():
        if turn.step > 0:
            received = turn.frame.data_structure("plant").entity_set("SubOrgan")
            log[turn.step] = dict(zip(received.columns["id"].tolist(),
                                      np.asarray(received.columns["water_uptake"]).tolist()))
            segments.append(segments[-1] + 1)
        count = len(segments)
        sets = [entity_set_of("Plant", 1, [plant], [NO_ID], [NO_ID], "Wheat"),
                entity_set_of("Organ", 5, [organ], [plant], [NO_ID], "RootInternode"),
                entity_set_of("SubOrgan", 6, segments, [organ] * count, [NO_ID] + segments[:-1], "RootSegment",
                            length=np.ones(count), radius=np.full(count, 0.05))]
        participant.send(encoding3d.Frame(data_structures=[encoding3d.DataStructure("plant", entity_sets=sets)]))


def entity_set_of(name, rank, ids, complexes, parents, label, **variables):
    columns = dict(id=np.array(ids, dtype=np.int64), complex=np.array(complexes, dtype=np.int64),
                   parent=np.array(parents, dtype=np.int64),
                   edge_type=np.array(["none" if parent == NO_ID else "successor" for parent in parents], dtype=object),
                   label=np.array([label] * len(ids), dtype=object), **variables)
    return encoding3d.EntitySet(name, rank=rank, columns=columns)


def test_metafspm_joins_a_plant_another_platform_builds():
    """join(): no plant of metafspm's; its MPG and DataStructure built from the session's first turn (another
    platform's plant, organ and segments), then its component computes on them, the new segments followed."""
    coordinator = Coordinator("tcp://127.0.0.1:*", steps=3, interval=1., expect=["roots", "metafspm"], timeout=120)
    log = {}
    threads = [threading.Thread(target=coordinator.run, daemon=True),
               threading.Thread(target=run_roots, args=(coordinator.endpoint, log), daemon=True)]
    for thread in threads:
        thread.start()
    data_structure = RemoteRoots.join(coordinator.endpoint)
    assert data_structure.n_nodes() == 2
    mtg = data_structure._mtg
    anchors = set(mtg.scales.anchors.values())
    assert len([v for v in mtg.vertices(scale=mtg.scales.Plant) if v not in anchors]) == 1      # the remote plant
    uptake, remote = WaterUptake(data_structure=data_structure), RemoteRoots(data_structure=data_structure)
    remote.start()
    assert np.allclose(data_structure.get("length"), 0.01)                          # cm converted to m
    for _ in range(3):
        uptake()
        expected = 1e-6 * 2 * np.pi * np.asarray(data_structure.get("radius")) * np.asarray(data_structure.get("length"))
        assert np.allclose(data_structure.get("water_uptake"), expected)
        remote()
    for thread in threads:
        thread.join(60)
    assert remote.stopped == "end of the session"
    assert data_structure.n_nodes() == 5                                             # one more segment at steps 1, 2 and 3
    assert log[2] and all(value == pytest.approx(1e-6 * 2 * np.pi * 0.0005 * 0.01) for value in log[2].values()
                          if value == value)
