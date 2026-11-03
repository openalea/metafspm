"""
metafspm in an fspm-TF coupling session (openalea.metafspm.io.coupling): the root growth model coupled to a carbon
model of another platform through a RemoteComponent, the session run by fspm-TF's coordinator.

The remote carbon model here is a plain fspm-TF participant in a thread (any platform's would answer the same
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

from openalea import fspm_tf  # noqa: E402
from openalea.fspm_tf.coupling import (Coordinator, Declaration, EntitySetSpec, Link, Participant,  # noqa: E402
                                       VariableSpec)
from openalea.metafspm.coupling.component import input_variable, state_variable  # noqa: E402
from openalea.metafspm.data_structure.configs import ScalesConfig as scales  # noqa: E402
from openalea.metafspm.io.coupling import RemoteComponent  # noqa: E402

from growth import DOC, RootGrowthProbe  # noqa: E402
from test_fspm_tf import _population  # noqa: E402

NO_ID = fspm_tf.NO_ID
STEPS = 4
# The command running Julia with FspmTF, PlantGraphs and ZMQ (e.g. "julia --project=env"): the carbon model in
# VirtualPlantLab (vpl_carbon.jl)
JULIA = os.environ.get("FSPM_TF_JULIA")
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
        participant.send(fspm_tf.Frame(data_structures=[fspm_tf.DataStructure("plant", entity_sets=[
            fspm_tf.EntitySet("Layer", rank=7, columns=columns)])], update=fspm_tf.Update("carbon", removed)))


def run_metafspm(endpoint_of, start_carbon, tmp_path, removal=True):
    """The growth model and the remote carbon model coupled for STEPS steps; checks what metafspm receives."""
    ds = _population()
    growth, remote = RootGrowthProbe(data_structure=ds), RemoteCarbon(data_structure=ds)
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
    assert np.allclose(ds.get("C_hexose_root"), 1.5)
    g = ds._mtg

    def layers():                                                   # the Layer vertices under segments (not anchors)
        nodes = set(ds.entity_ids("node").tolist())
        return [v for v in g.vertices(scale=g.scales.Layer) if g.complex(v) in nodes]

    assert len(layers()) == 2 * ds.n_nodes()                        # followed: created in metafspm's MPG
    assert {g.complex(v) for v in layers()} == set(ds.entity_ids("node").tolist())
    nodes = ds.n_nodes()
    for step in range(1, STEPS + 1):
        before = set(ds.entity_ids("node").tolist())
        growth()
        masses = np.array(ds.get("struct_mass"), dtype=float)
        new = np.array([vid not in before for vid in ds.entity_ids("node").tolist()])
        remote()
        values = np.asarray(ds.get("C_hexose_root"))
        if step == 3 and removal:        # the segment that lost its first layer: the mean of its other, 2 / (1 + m)
            differs = ~np.isclose(values, expected(masses, new))
            assert differs.sum() == 1 and np.isclose(values[differs], 2. / (1. + masses[differs])).all()
        elif step < STEPS:
            assert np.allclose(values, expected(masses, new)), step
    assert remote.stopped == "end of the session"
    assert ds.n_nodes() > nodes                                     # the roots grew new segments, sent each step
    # the layers of the segments before the last step (those of the last step's new segments come with a turn the
    # session no longer sends: they are in its record), one removed by the carbon model at step 3
    assert len(layers()) == 2 * len(before) - (1 if removal else 0)
    session.join(60)
    for thread in threads:
        thread.join(60)
    record = fspm_tf.read_index(str(tmp_path / "record" / "scene.fspm.json"))
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


@pytest.mark.skipif(not JULIA, reason="FSPM_TF_JULIA (Julia with FspmTF, PlantGraphs and ZMQ) is not set")
@pytest.mark.parametrize("transport", sorted(ENDPOINTS))
def test_virtualplantlab_as_the_carbon_model(transport, tmp_path):
    """The same coupling, the carbon model in VirtualPlantLab: its layers a PlantGraphs graph grown by a rule."""
    def start_carbon(coordinator, log):
        coordinator.launch("carbon", f"{JULIA} {VPL_CARBON}")
        return []

    run_metafspm(ENDPOINTS[transport], start_carbon, tmp_path, removal=False)
