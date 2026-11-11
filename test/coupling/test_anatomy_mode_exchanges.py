"""
Exchanges with an anatomy-mode plant (its nodes the Compartments of its SubOrgans' anatomies, its segments' ends at
its SubOrgans): a light-like model on the union of its SubOrgans gives its values to the plant's SubOrgan-scale
component (on the plant's view at its SubOrgans) when the scene exchanges into the plant's DataStructure; a soil
grid's CrossMapping maps its SubOrgans, a Compartment variable going through its SubOrgan's cells, its masked
Compartments only (the roots' soil nodes).
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.cross import CrossMapping, Exchanges, UnionDataStructure, UnionMapping
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, LocatedView, MPGDataStructure
from openalea.metafspm.solve.decorator import rate

from anatomy import make_rooted_anatomy, wiring
from growth import DOC

N_SEGMENTS, CELL = 3, 0.4                # vertical root segments, each in one soil cell of 0.4
SURFACE = 1.                             # the compartment type of a segment's soil node (its epidermis here)


def _descriptor():
    return state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")


@dataclass
class RootEnds(FunctionalComponent):
    """The segments' ends (an architecture's, at the SubOrgans) and the light they receive."""
    x1: float = _descriptor()
    x2: float = _descriptor()
    y1: float = _descriptor()
    y2: float = _descriptor()
    z1: float = _descriptor()
    z2: float = _descriptor()
    intercepted: float = input_variable(**DOC, by="Lamp", default=0., scale=scales.SubOrgan)


@dataclass
class Cells(FunctionalComponent):
    """At the Compartments: the soil nodes' uptake (extensive) and the soil's potential they receive."""
    compartment_type: float = state_variable(**DOC, default=0., location="node", kind="descriptor")
    uptake: float = state_variable(**DOC, default=0., location="node", kind="extensive")
    soil_potential: float = input_variable(**DOC, by="Soil", default=0., location="node")


@dataclass
class Lamp(FunctionalComponent):
    """On the union of the plants' SubOrgans: each one's light, its rank plus one."""
    intercepted: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="intensive")

    @rate
    def _intercepted(self, intercepted):
        return np.arange(intercepted.size) + 1.


@dataclass
class Soil(FunctionalComponent):
    uptake: float = input_variable(**DOC, by="Cells", default=0., location="cell", kind="extensive")
    potential: float = state_variable(**DOC, default=0., location="cell", kind="intensive")


def _plant():
    """Three vertical root segments, from the surface down, each with its anatomy (four Compartments, its epidermis
    the soil node: compartment_type SURFACE)."""
    mtg, segments, anatomies = make_rooted_anatomy(n_segments=N_SEGMENTS)
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg))
    ends, cells = RootEnds(data_structure=data_structure), Cells(data_structure=data_structure)
    depth = CELL * np.arange(N_SEGMENTS)
    values = dict(x1=0.2, x2=0.2, y1=0.2, y2=0.2, z1=-depth, z2=-(depth + CELL))
    for name, value in values.items():
        data_structure.set(name, np.broadcast_to(value, (N_SEGMENTS,)).astype(float))
    compartment_type = np.zeros(data_structure.n_nodes())
    compartment_type[data_structure.index_of([anatomy["epidermis"] for anatomy in anatomies.values()])] = SURFACE
    data_structure.set("compartment_type", compartment_type)
    data_structure.define_mask("soil_nodes", {"compartment_type": SURFACE})
    return data_structure, ends, cells, segments


def test_a_union_s_values_reach_a_suborgan_component_of_an_anatomy_mode_plant():
    data_structure, ends, _, _ = _plant()
    assert isinstance(ends.data_structure, LocatedView)
    union = UnionDataStructure([data_structure.at("SubOrgan")])
    lamp = Lamp(data_structure=union)
    exchanges = Exchanges(Translator().link("RootEnds", "intercepted", "Lamp", {"intercepted": 1.}),
                          (ends, lamp), (UnionMapping(union),))
    lamp()
    exchanges.exchange(into=data_structure)                   # into the plant's DataStructure, as the Scene's
    np.testing.assert_allclose(ends.data_structure.get("intercepted"), [1., 2., 3.])


def test_a_soil_mapping_maps_the_suborgans_and_their_masked_compartments():
    data_structure, ends, cells, segments = _plant()
    soil_grid = ArrayDataStructure(shape=(1, 1, N_SEGMENTS + 1), spacing=CELL)
    soil = Soil(data_structure=soil_grid)
    mapping = CrossMapping(data_structure, soil_grid, method="overlap", mask="soil_nodes")
    assert mapping.location == "SubOrgan"
    rows, columns, weights = mapping.incidence()
    assert rows.tolist() == [0, 1, 2] and columns.tolist() == [0, 1, 2]
    np.testing.assert_allclose(weights, 1.)
    soil_nodes = np.flatnonzero(np.asarray(data_structure.get("compartment_type")) == SURFACE)
    uptake = np.zeros(data_structure.n_nodes())
    uptake[soil_nodes] = [1., 2., 4.]
    uptake[soil_nodes[0] + 1] = 100.              # a Compartment other than the soil nodes: not exchanged
    data_structure.set("uptake", uptake)
    soil_grid.set("potential", np.reshape([-1., -2., -3., -4.], soil_grid.get("potential").shape))
    translator = (Translator().link("Soil", "uptake", "Cells", {"uptake": 1.})
                  .link("Cells", "soil_potential", "Soil", {"potential": 1.}))
    exchanges = Exchanges(translator, (cells, soil), (mapping,))
    exchanges.exchange(into=soil_grid)
    np.testing.assert_allclose(np.ravel(soil_grid.get("uptake")), [1., 2., 4., 0.])
    exchanges.exchange(into=data_structure)
    potential = np.asarray(data_structure.get("soil_potential"))
    np.testing.assert_allclose(potential[soil_nodes], [-1., -2., -3.])
    np.testing.assert_array_equal(np.delete(potential, soil_nodes), 0.)
