"""
A reduced plant's sections drawn whole in an encoding3d frame: a component's display_triangles() draws a reduced
section's interior cells too (owner -1, owner_ids their display ids: solve.reduction.display_ids), and the frame writes
the interiors as display rows (metafspm_display_only) of their scales' entity sets: the reduced system's unknown
recovered (Reduction.recover_interiors, one factorisation per archetype) equal to the full plant's solve, their template's
properties, linked to their Cells, Layers and section; the mesh's triangles on them. from_frame leaves them out (the
DataStructure stays reduced); a selection by entity brings a section's interior with it.
"""
import numpy as np
import pytest

from openalea import encoding3d
from openalea.encoding3d.validate import validate_bytes

from openalea.metafspm.coupling.geometry import Triangles
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.io.encoding3d import DISPLAY_ONLY, filter_frame, from_frame, to_frame
from openalea.metafspm.solve.reduction import display_ids, is_display_id

from layered_anatomy import section_template
from test_reduction import _places, _solved


class SectionDrawer:
    """A display component drawing one triangle per Compartment of every section, a reduced one's whole (its kept
    nodes by node index, its interior by display id)."""

    def __init__(self, data_structure):
        self.data_structure = data_structure

    def display_triangles(self, where=None):
        data_structure = self.data_structure
        mtg = data_structure.structure
        owners, ids = list(range(data_structure.n_nodes())), list(data_structure.entity_ids("node"))
        reduction = mtg.reduction
        for owner, key in (reduction.reduced_groups(data_structure).values() if reduction is not None else ()):
            full = reduction.full_template(key)
            held = set(reduction.full_rows(key).tolist())
            interior = [row for row in np.flatnonzero(full.scale == scales.Compartment).tolist() if row not in held]
            owners += [-1] * len(interior)
            ids += display_ids(owner, interior).tolist()
        n = len(owners)
        corners = np.stack([np.zeros((n, 3)), np.eye(3)[[0] * n], np.eye(3)[[1] * n]], 1) + np.arange(n)[:, None, None]
        return Triangles(corners, owners, ids)


@pytest.fixture(scope="module")
def solved():
    full, reduced = _solved(False), _solved(True)
    return full.data_structure, reduced.data_structure, reduced.instance


def _frame(reduced, instance):
    return to_frame({"plants": reduced}, components={"plants": list(instance.components) + [SectionDrawer(reduced)]},
                    display="full")


def test_the_interiors_are_display_rows_with_the_full_solve_s_values(solved):
    full, reduced, instance = solved
    frame = _frame(reduced, instance)
    plants = frame.data_structures[0]
    compartments = plants.entity_set("Compartment").columns
    shown = np.asarray(compartments[DISPLAY_ONLY], dtype=bool)
    assert shown.sum() + reduced.n_nodes() == full.n_nodes() and is_display_id(compartments["id"][shown]).all()
    np.testing.assert_array_equal(compartments["id"][~shown], reduced.entity_ids("node"))
    # the recovered potentials: the full plant's at the same places (section, full template row)
    template = section_template()[0]
    row_in_full = {place: i for i, place in enumerate(_places(full))}
    sections = reduced.entity_ids("SubOrgan").tolist()
    expected = []
    for identifier in compartments["id"][shown].tolist():
        owner, row = divmod(identifier - display_ids(0, [0])[0], 2 ** 24)
        expected.append(full.get("water_potential")[row_in_full[(sections.index(owner), row)]])
        assert template.scale[row] == scales.Compartment
    np.testing.assert_allclose(compartments["water_potential"][shown], expected, rtol=1e-11)
    np.testing.assert_allclose(compartments["water_potential"][~shown], reduced.get("water_potential"))
    # their template's properties (archetype), linked to their Cells (themselves display rows) and section
    assert np.isfinite(compartments["compartment_type"][shown]).all()
    cells = plants.entity_set("Cell").columns
    assert np.asarray(cells[DISPLAY_ONLY], dtype=bool).sum() == 3 * len(sections)      # the cortex's cells
    assert np.isin(compartments["complex"][shown], np.concatenate([cells["id"], np.asarray(sections)])).all()
    assert validate_bytes(encoding3d.to_bytes(frame)).ok


def test_the_mesh_draws_the_interiors_on_their_rows(solved):
    _, reduced, instance = solved
    plants = _frame(reduced, instance).data_structures[0]
    mesh = next(mesh for mesh in plants.meshes if mesh.name == "SectionDrawer")
    triangles = SectionDrawer(reduced).display_triangles()
    compartments = plants.entity_set("Compartment").columns
    rows = mesh.feature_rows["Compartment"][::3]
    np.testing.assert_array_equal(compartments["id"][rows], triangles.owner_ids)
    sections = plants.entity_set("SubOrgan").columns["id"][mesh.feature_rows["SubOrgan"][::3]]
    owner_of = {int(v): int(s) for v, s in zip(reduced.entity_ids("node"),
                                               reduced.entity_ids("SubOrgan")[reduced.owner("SubOrgan")])}
    for identifier, section in zip(triangles.owner_ids.tolist(), sections.tolist()):
        owner = (identifier - display_ids(0, [0])[0]) // 2 ** 24 if is_display_id([identifier])[0] else \
            owner_of[identifier]
        assert section == owner
    assert (mesh.feature_rows["Cell"][::3][triangles.owner < 0] >= 0).any()


def test_from_frame_rebuilds_the_reduced_plant_and_a_selection_brings_a_section_s_interior(solved):
    _, reduced, instance = solved
    frame = _frame(reduced, instance)
    rebuilt = from_frame(frame)["plants"]
    assert rebuilt.n_nodes() == reduced.n_nodes()
    np.testing.assert_array_equal(rebuilt.entity_ids("node"), reduced.entity_ids("node"))
    np.testing.assert_allclose(rebuilt.get("water_potential"), reduced.get("water_potential"))
    section = int(reduced.entity_ids("SubOrgan")[1])
    filter_frame(frame, entities=[section])
    compartments = frame.data_structures[0].entity_set("Compartment").columns
    shown = np.asarray(compartments[DISPLAY_ONLY], dtype=bool)
    owners = (compartments["id"][shown] - display_ids(0, [0])[0]) // 2 ** 24
    assert shown.any() and (owners == section).all()


def test_a_variables_filter_applies_to_the_display_rows(solved):
    _, reduced, instance = solved
    full = _frame(reduced, instance).data_structures[0].entity_set("Compartment").columns
    frame = filter_frame(_frame(reduced, instance),
                         variables={"plants": {"Compartment": ["water_potential"], "*": []}})
    compartments = frame.data_structures[0].entity_set("Compartment").columns
    assert set(compartments) == {"id", "label", "complex", "parent", "edge_type", "water_potential", DISPLAY_ONLY}
    np.testing.assert_array_equal(compartments[DISPLAY_ONLY], full[DISPLAY_ONLY])
    np.testing.assert_array_equal(compartments["water_potential"], full["water_potential"])
    assert set(frame.data_structures[0].entity_set("Cell").columns) <= {"id", "label", "complex", "parent",
                                                                        "edge_type", DISPLAY_ONLY}
    assert validate_bytes(encoding3d.to_bytes(frame)).ok


def test_without_a_display_component_no_display_row_is_written(solved):
    _, reduced, instance = solved
    compartments = to_frame({"plants": reduced}, components={"plants": list(instance.components)}) \
        .data_structures[0].entity_set("Compartment").columns
    assert DISPLAY_ONLY not in compartments
