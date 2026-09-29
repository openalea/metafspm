"""
Plant graph <-> soil grid Coupler (devplan WD.5, design note §7): vertex -> voxel map from segment barycenters in the
(x, y, z) soil grid, extensive scatter-sum to the soil, intensive gather to the plant.
"""
import numpy as np
import pytest

import doubles
import doubles_ds
from openalea.metafspm.coupling.coupler import Coupler, VoxelLocator
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import ArrayDataStructure

SIDE = doubles.SOIL_VOXEL_SIDE


def _plant_with_geometry(x=0.025, y=0.075):
    """Seedling plant DataStructure: node i is a vertical segment at (x, y), depth 0.01 + 0.02 i to 0.03 + 0.02 i."""
    ds = doubles_ds.make_plant_ds()
    n = ds.n_nodes()
    top = -0.01 - 0.02 * np.arange(n)
    for name, values in (("x1", x), ("x2", x), ("y1", y), ("y2", y), ("z1", top), ("z2", top - 0.02)):
        ds.register(name, values, location="node", on_grow="inherit")
    ds.register("hexose_exudation", np.arange(n, dtype=float) + 1., location="node")
    ds.register("amino_acids_exudation", 0.5, location="node")
    ds.register("C_hexose_soil", location="node")
    ds.register("soil_temperature", location="node")
    return ds


def _soil(nx=2, ny=2, nz=4):
    soil = ArrayDataStructure(shape=(nx, ny, nz), dx=SIDE)
    for name in ("hexose_exudation_massic", "amino_acids_exudation"):
        soil.register(name)
    soil.register("C_hexose_soil", np.arange(nx * ny * nz, dtype=float).reshape(nx, ny, nz))
    soil.register("soil_temperature", 10.)
    return soil


def _coupler(plant, soil):
    return Coupler(plant, soil, VoxelLocator(soil, periodic=(True, True, False), flip_z=True),
                   to_soil={"hexose_exudation_massic": {"hexose_exudation": 72.},
                            "amino_acids_exudation": {"amino_acids_exudation": 5.}},
                   to_plant={"C_hexose_soil": "C_hexose_soil"})


def _depth_layer(n):
    return np.floor((0.02 + 0.02 * np.arange(n)) / SIDE).clip(0, 3).astype(int)


def test_vertex_to_voxel_map():
    plant, soil = _plant_with_geometry(), _soil()
    coupler = _coupler(plant, soil)
    coupler.update_map()
    ix, iy, iz = np.unravel_index(coupler.cells, soil.shape)
    assert (ix == 0).all() and (iy == 1).all()
    np.testing.assert_array_equal(iz, _depth_layer(plant.n_nodes()))


def test_push_sums_extensive_fluxes_with_factors():
    plant, soil = _plant_with_geometry(), _soil()
    coupler = _coupler(plant, soil)
    coupler.update_map()
    coupler.zero_soil_inputs()
    coupler.push()

    layers = _depth_layer(plant.n_nodes())
    massic = soil.get("hexose_exudation_massic")
    for layer in range(4):
        members = layers == layer
        assert massic[0, 1, layer] == pytest.approx(72. * plant.get("hexose_exudation")[members].sum())
        assert soil.get("amino_acids_exudation")[0, 1, layer] == pytest.approx(5. * 0.5 * members.sum())
    assert massic[1].sum() == 0. and massic[:, 0].sum() == 0.


def test_several_plants_add_up_after_one_zeroing():
    soil = _soil()
    first, second = _plant_with_geometry(), _plant_with_geometry(x=0.075, y=0.075)
    couplers = [_coupler(first, soil), _coupler(second, soil)]
    for c in couplers:
        c.update_map()
    couplers[0].zero_soil_inputs()
    for c in couplers:
        c.push()
    massic = soil.get("hexose_exudation_massic")
    assert massic[0].sum() == pytest.approx(72. * first.get("hexose_exudation").sum())
    assert massic[1].sum() == pytest.approx(72. * second.get("hexose_exudation").sum())


def test_pull_gathers_intensive_states_in_place():
    plant, soil = _plant_with_geometry(), _soil()
    coupler = _coupler(plant, soil)
    view = plant.get("C_hexose_soil")
    coupler.update_map()
    coupler.pull()
    assert plant.get("C_hexose_soil") is view
    np.testing.assert_array_equal(view, soil.get("C_hexose_soil").ravel()[coupler.cells])


def test_periodic_xy_and_clipped_depth():
    plant, soil = _plant_with_geometry(x=0.125, y=-0.025), _soil()   # x wraps to 0.025, y to 0.075
    coupler = _coupler(plant, soil)
    coupler.update_map()
    ix, iy, iz = np.unravel_index(coupler.cells, soil.shape)
    assert (ix == 0).all() and (iy == 1).all() and iz.max() == 3   # deep segments clipped to the last layer


def test_map_follows_growth():
    plant, soil = _plant_with_geometry(), _soil()
    coupler = _coupler(plant, soil)
    coupler.update_map()
    g = plant.mtg
    from simple_seedling import generate_simple_mpg_seedling  # noqa: F401 (path set by doubles_ds)
    tip = next(v for v in plant._idx_to_vid if not any(p == v for p, _ in plant.edges()))
    g.add_child(tip, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment))
    plant.update_topology()

    coupler.update_map()      # rebuilt for the new topology
    assert coupler.cells.size == plant.n_nodes()
    coupler.zero_soil_inputs()
    coupler.push()
    assert soil.get("hexose_exudation_massic").sum() == pytest.approx(72. * plant.get("hexose_exudation").sum())


def test_push_before_map_update_after_growth_raises():
    plant, soil = _plant_with_geometry(), _soil()
    coupler = _coupler(plant, soil)
    coupler.update_map()
    g = plant.mtg
    tip = next(v for v in plant._idx_to_vid if not any(p == v for p, _ in plant.edges()))
    g.add_child(tip, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment))
    plant.update_topology()
    with pytest.raises(RuntimeError, match="update_map"):
        coupler.push()


def test_matches_the_reference_soil_model():
    """Same sums as rhizosoil's apply_to_voxel_fast / get_from_voxel_fast (legacy (y, z, x) voxel arrays)."""
    plant = _plant_with_geometry()
    n = plant.n_nodes()
    soil = _soil(nx=2, ny=2, nz=2)                  # the reference double is 0.1 x 0.1 x 0.1 m
    legacy = doubles.SoilModel()
    legacy.pullable_inputs = {"P": {"hexose_exudation_massic": {"hexose_exudation": 72.},
                                    "amino_acids_exudation": {"amino_acids_exudation": 5.}}}
    hs = {name: row for row, name in enumerate(["vertex_index", "x1", "x2", "y1", "y2", "z1", "z2",
                                                  "hexose_exudation", "amino_acids_exudation"])}
    buf = np.zeros((len(hs), n))
    buf[hs["vertex_index"]] = 1.
    for name in hs:
        if name != "vertex_index":
            buf[hs[name]] = plant.get(name)
    mask = buf[hs["vertex_index"]] >= 1
    for name in legacy.inputs:
        legacy.voxels[name].fill(0.)
    iy, iz, ix = legacy.compute_mtg_voxel_neighbors_fast(buf, hs, mask=mask, flip_z=True)
    legacy.apply_to_voxel_fast(iy, iz, ix, buf, hs, "P", mask)

    coupler = _coupler(plant, soil)
    coupler.update_map()
    coupler.zero_soil_inputs()
    coupler.push()

    for name in ("hexose_exudation_massic", "amino_acids_exudation"):
        np.testing.assert_allclose(soil.get(name), legacy.voxels[name].transpose(2, 0, 1))   # (y, z, x) -> (x, y, z)


def test_coupler_from_translator():
    translator = Translator.from_dict(doubles_ds.translator())
    plant, soil = _plant_with_geometry(), _soil()
    coupler = Coupler.from_translator(translator, plant_components=["PlantCarbon", "PlantNitrogen"], soil="SoilModel",
                                      plant_ds=plant, soil_ds=soil, locator=VoxelLocator(soil, flip_z=True))
    assert coupler.to_soil == {"hexose_exudation_massic": {"hexose_exudation": 72.},
                               "amino_acids_exudation": {"amino_acids_exudation": 5.}}
    assert coupler.to_plant == {"soil_temperature": "soil_temperature", "C_hexose_soil": "C_hexose_soil"}
    assert coupler.plant_variables() == ["x1", "x2", "y1", "y2", "z1", "z2", "hexose_exudation", "amino_acids_exudation"]
    assert coupler.soil_variables() == ["soil_temperature", "C_hexose_soil"]
