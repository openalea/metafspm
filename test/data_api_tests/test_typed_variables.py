"""
Typed variables (design note time_and_data §5, step 4c, DS12, T6, Q22): integer variables kept as integers, label
names resolved through the MTG's LabelsConfig, and object variables (lists, records) stored per entity but kept out
of graph systems, derivations and transport.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import _type_mask, graph_system, node_balance, rate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structure_tests'))
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])


@pytest.fixture(autouse=True)
def _fresh_choregrapher():
    Choregrapher().reset()
    yield
    Choregrapher().reset()


@pytest.fixture
def seedling():
    g, s = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return g, s, MPGDataStructure(g, from_scale=g.scales.SubOrgan)


@dataclass
class Labelled(FunctionalComponent):
    label: int = parameter(**DOC, by="", default=0, scale=scales.SubOrgan, dtype="int")


def test_integer_labels_are_read_and_kept_as_integers(seedling):
    g, _, ds = seedling
    Labelled(data_structure=ds)
    labels = ds.get("label")
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels.tolist() == [g.property("label")[int(v)] for v in ds.entity_ids("node")]
    ds.set("label", labels)                                  # integral values are fine
    with pytest.raises(ValueError, match="holds integers"):
        ds.set("label", 1.5)


def test_label_names_in_masks_and_filters(seedling):
    g, _, ds = seedling
    Labelled(data_structure=ds)
    ds.define_mask("roots", {"label": ["RootSegment"]})
    ds.define_mask("leaves", {"label": "LeafElement"})
    labels = ds.get("label")
    np.testing.assert_array_equal(ds.mask("roots"), labels == g.labels.SubOrgan.RootSegment)
    np.testing.assert_array_equal(ds.mask("leaves"), labels == g.labels.SubOrgan.LeafElement)
    snap = {"label": labels.astype(float)}
    np.testing.assert_array_equal(_type_mask({"label": ["RootSegment", "StemElement"]}, snap, ds.n_nodes(), ds),
                                  np.isin(labels, [g.labels.SubOrgan.RootSegment, g.labels.SubOrgan.StemElement]))


def test_label_names_follow_the_scale_of_the_variable_and_raise_when_ambiguous():
    from anatomy import make_rooted_anatomy, wiring
    g, _, _ = make_rooted_anatomy(2)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))

    @dataclass
    class CellLabels(FunctionalComponent):
        label: int = parameter(**DOC, by="", default=0, scale=scales.Compartment, dtype="int")

    CellLabels(data_structure=ds)
    assert ds.label_code("Symplastic", "label") == g.labels.Compartment.Symplastic   # its own scale's group first
    with pytest.raises(ValueError, match=r"'Symplastic' is ambiguous: \['Compartment.Symplastic', 'Connection.Symplastic'\]"):
        ds.label_code("Symplastic")
    with pytest.raises(ValueError, match="unknown label 'Bark'"):
        ds.label_code("Bark")
    assert ds.label_code("SymplasticNode") == g.labels.Compartment.Symplastic        # label values are unique


@dataclass
class Vessels(FunctionalComponent):
    vessel_radii: object = state_variable(**DOC, initialize=None, scale=scales.SubOrgan, dtype="object")
    n_vessels: int = state_variable(**DOC, initialize=0, scale=scales.SubOrgan, dtype="int")

    @rate(vectorized=False)
    def _n_vessels(self, vessel_radii):
        return len(vessel_radii) if vessel_radii is not None else 0


def test_object_variables_are_stored_per_entity_and_round_trip_through_the_mtg(seedling):
    g, s, ds = seedling
    radii = {int(v): [1e-5 * (i + 1)] * (i % 3 + 1) for i, v in enumerate(ds.entity_ids("node"))}
    g.properties()["vessel_radii"] = dict(radii)
    model = Vessels(data_structure=ds)
    assert ds.get("vessel_radii").dtype == object
    model()
    counts = dict(zip(ds.entity_ids("node").tolist(), ds.get("n_vessels").tolist()))
    assert counts == {v: len(r) for v, r in radii.items()}
    assert dict(g.property("vessel_radii")) == radii                  # written back by identity
    new = g.add_child(s.root_segment6, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                     label=g.labels.SubOrgan.RootSegment))
    ds.update_topology()
    by_vid = dict(zip(ds.entity_ids("node").tolist(), ds.get("vessel_radii").tolist()))
    assert by_vid[new] is None and by_vid[int(s.root_segment6)] == radii[int(s.root_segment6)]


@dataclass
class UsesObjects(FunctionalComponent):
    c: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    vessel_radii: object = parameter(**DOC, by="", default=None, scale=scales.SubOrgan, dtype="object")

    @graph_system(node_unknowns=["c"], solver="newton")
    class _solve:
        @node_balance(field="c")
        def _balance(self, c, vessel_radii):
            return c


def test_object_variables_are_rejected_by_graph_systems_and_derivations(seedling):
    _, _, ds = seedling
    model = UsesObjects(data_structure=ds)
    with pytest.raises(TypeError, match="'vessel_radii' holds objects, which graph systems cannot use"):
        model()
    with pytest.raises(TypeError, match="holds objects, which cannot be derived"):
        ds.derive("radii_copy", {"vessel_radii": 1.})
