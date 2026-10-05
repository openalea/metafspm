"""
Persistence: a DataStructure checkpointed and restored continues bit for bit;
arrays in npz, a JSON manifest, and a pickle for objects, formulas, mask rules and the MPG.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.scene.population import build_population
from openalea.metafspm.solve.decorator import rate
from growth import DOC, CarbonProbe, RootGrowthProbe



def _population():
    table = pd.DataFrame([dict(plant=f"p{i}", model=None, x=0.1 * i, y=0., z=0., rotation=0.,
                               scenario={"parameters": {"n_segments": 2 + i}}) for i in range(3)])
    g, _ = build_population(table, initiators=(RootGrowthProbe,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def _models(ds):
    return RootGrowthProbe(data_structure=ds), CarbonProbe(data_structure=ds)


def _steps(models, n):
    for _ in range(n):
        for model in models:
            model()


def _hexose_ratio(hexose, length):           # a module-level formula: kept by reference
    return hexose / length


def _state(ds):
    return {location: {name: np.array(values) for name, values in store.items()}
            for location, store in ds._var_stores().items()}


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for location in a:
        assert a[location].keys() == b[location].keys(), location
        for name in a[location]:
            np.testing.assert_array_equal(a[location][name], b[location][name], err_msg=f"{location}.{name}")


def test_a_restored_population_continues_bit_for_bit(tmp_path):
    ds = _population()
    models = _models(ds)
    ds.derive("hexose_per_length", ("C_hexose_root", "length"), formula=_hexose_ratio)
    ds.define_mask("apices", {"is_apex": ">0"})
    _steps(models, 2)
    ds.checkpoint(str(tmp_path / "checkpoint"))
    _steps(models, 2)                                         # uninterrupted
    expected = _state(ds)

    Choregrapher().reset()
    restored = MPGDataStructure.restore(str(tmp_path / "checkpoint"))
    assert restored.mtg is not ds.mtg and restored.has("hexose_per_length") and restored.has_mask("apices")
    _steps(_models(restored), 2)                              # restarted
    _assert_same(_state(restored), expected)
    np.testing.assert_array_equal(restored.get("hexose_per_length"), ds.get("hexose_per_length"))
    np.testing.assert_array_equal(restored.mask("apices"), ds.mask("apices"))
    assert restored.topology_version == ds.topology_version


def test_the_checkpoint_layout(tmp_path):
    ds = _population()
    _models(ds)
    ds.register("tags", [["a"]] * ds.n_nodes(), location="node", dtype=object)
    ds.checkpoint(str(tmp_path / "c"))
    assert sorted(os.listdir(tmp_path / "c")) == ["arrays.npz", "manifest.json", "state.pkl"]
    manifest = json.loads((tmp_path / "c" / "manifest.json").read_text())
    assert manifest["class"] == "MPGDataStructure" and manifest["variables"]["length"] == {"location": "node",
                                                                                         "dtype": "float64"}
    assert manifest["variables"]["tags"]["dtype"] == "object"
    with np.load(tmp_path / "c" / "arrays.npz") as npz:
        assert "node::length" in npz.files and "node::tags" not in npz.files       # objects are pickled
    restored = MPGDataStructure.restore(str(tmp_path / "c"))
    assert list(restored.get("tags")) == [["a"]] * ds.n_nodes()


def test_the_mtg_can_be_given_back(tmp_path):
    ds = _population()
    _models(ds)
    ds.checkpoint(str(tmp_path / "c"), include_mtg=False)
    with pytest.raises(ValueError, match="holds no MTG"):
        MPGDataStructure.restore(str(tmp_path / "c"))
    restored = MPGDataStructure.restore(str(tmp_path / "c"), mtg=ds.mtg)
    np.testing.assert_array_equal(restored.get("length"), ds.get("length"))
    with pytest.raises(ValueError, match="differ from the checkpointed ones"):
        MPGDataStructure.restore(str(tmp_path / "c"), mtg=_grown(ds))


def _grown(ds):
    import pickle
    g = pickle.loads(pickle.dumps(ds.mtg))
    other = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    growth = RootGrowthProbe(data_structure=other)
    for _ in range(4):
        growth()
    return other.mtg


def test_lambdas_are_refused_with_a_hint(tmp_path):
    ds = _population()
    _models(ds)
    ds.derive("twice", {"length": 2.})
    ds.define_mask("long", lambda d: d.get("length") > 0.5)
    with pytest.raises(TypeError, match="module-level functions"):
        ds.checkpoint(str(tmp_path / "c"))


@dataclass
class GridDiffusionProbe(FunctionalComponent):
    nitrate: float = state_variable(**DOC, initialize=1., location="cell")

    @rate
    def _nitrate(self, nitrate):
        return 0.9 * nitrate + 0.01 * nitrate.sum()


def test_a_restored_grid_continues_bit_for_bit(tmp_path):
    grid = ArrayDataStructure(shape=(3, 2, 2), dx=(0.1, 0.2, 0.05), origin=(1., 0., 0.), periodic=(True, False, False))
    model = GridDiffusionProbe(data_structure=grid)
    grid.set("nitrate", np.arange(12.).reshape(3, 2, 2))
    model()
    grid.checkpoint(str(tmp_path / "grid"))
    model()
    model()
    Choregrapher().reset()
    restored = ArrayDataStructure.restore(str(tmp_path / "grid"))
    again = GridDiffusionProbe(data_structure=restored)
    again()
    again()
    np.testing.assert_array_equal(restored.get("nitrate"), grid.get("nitrate"))
    assert restored.periodic == grid.periodic and np.array_equal(restored.cell_centers(), grid.cell_centers())


def test_a_restored_anatomy_keeps_rewiring_incrementally(tmp_path):
    from anatomy import grow_segment, make_rooted_anatomy, wiring
    g, segments, _ = make_rooted_anatomy(3)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
    ds.register("water", np.arange(ds.n_nodes(), dtype=float), location="node", on_grow="inherit")
    ds.register("flux", np.arange(ds.n_edges(), dtype=float), location="edge")
    ds.checkpoint(str(tmp_path / "anatomy"))
    restored = MPGDataStructure.restore(str(tmp_path / "anatomy"))
    np.testing.assert_array_equal(restored.entity_ids("edge"), ds.entity_ids("edge"))
    for original in (ds, restored):              # the same growth on both MTGs
        grow_segment(original._mtg, max(original._mtg.vertices(scale=original._mtg.scales.SubOrgan)))
        original.update_topology()
    assert restored.rewired == ds.rewired
    np.testing.assert_array_equal(restored.entity_ids("edge"), ds.entity_ids("edge"))
    for name in ("water", "flux"):
        np.testing.assert_array_equal(restored.get(name), ds.get(name))
