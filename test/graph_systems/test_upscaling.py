"""
Upscaled variables: their declaration (state_variable(upscaling=upscaling(...))) and the network operator models
compute them with (solve.network.equivalent_conductance).
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable, upscaling
from openalea.metafspm.solve.network import equivalent_conductance


def _view(edges, n):
    """A GraphView-like object of *edges* [(tail, head)]: B[tail, e] = +1, B[head, e] = -1."""
    rows, columns, values = [], [], []
    for e, (tail, head) in enumerate(edges):
        rows += [tail, head]
        columns += [e, e]
        values += [1., -1.]
    return SimpleNamespace(incidence=csr_matrix((values, (rows, columns)), shape=(n, len(edges))))


def test_conductances_in_series_and_in_parallel():
    chain = _view([(0, 1), (1, 2)], 3)
    conductance, inflow = equivalent_conductance(chain, [2., 3.], sources=[0], sinks=[2])
    assert conductance == pytest.approx(1. / (1. / 2. + 1. / 3.))
    np.testing.assert_allclose(inflow, [conductance, 0., 0.])
    branches = _view([(0, 1), (0, 1), (1, 2)], 3)                        # two edges in parallel, then one
    conductance, _ = equivalent_conductance(branches, [1., 1., 2.], sources=[0], sinks=[2])
    assert conductance == pytest.approx(1. / (1. / 2. + 1. / 2.))


def test_sources_joined_through_their_own_conductance_and_their_shares():
    """A root-like tree: two surface nodes (Robin, as the soil around a root), joined to a collar node (the sink);
    each one's inflow is its share, the standard uptake fractions summing to 1."""
    tree = _view([(0, 1), (1, 2), (1, 3)], 4)                            # 0 the collar, 2 and 3 the surface
    soil = np.array([0., 0., 1., 4.])
    conductance, inflow = equivalent_conductance(tree, [10., 1., 1.], sources=soil > 0, sinks=[0],
                                                 source_conductance=soil)
    branch = [1. / (1. / 1. + 1. / 1.), 1. / (1. / 4. + 1. / 1.)]       # surface then edge, per branch
    assert conductance == pytest.approx(1. / (1. / sum(branch) + 1. / 10.))
    np.testing.assert_allclose(inflow[2:] / conductance, np.array(branch) / sum(branch))
    assert inflow[:2].tolist() == [0., 0.] and (inflow / conductance).sum() == pytest.approx(1.)


def test_a_piece_without_a_source_or_a_sink_carries_no_flow():
    graph = _view([(0, 1), (2, 3)], 4)                                   # 2-3: a floating piece
    conductance, inflow = equivalent_conductance(graph, [5., 5.], sources=[0], sinks=[1])
    assert conductance == pytest.approx(5.) and inflow[2:].tolist() == [0., 0.]
    with pytest.raises(ValueError, match="both a source and a sink"):
        equivalent_conductance(graph, [5., 5.], sources=[0], sinks=[0, 1])


@dataclass
class RootSystem(FunctionalComponent):
    root_system_conductance: float = state_variable(
        unit="cm3 hPa-1 d-1", description="the root system's conductance, Krs", default=0., location="scalar",
        upscaling=upscaling(of=("wall_conductance", "membrane_conductance"), law="network",
                            protocol="uniform soil water potential at the epidermis, the collar's xylem potential "
                                     "fixed: Krs = Q_collar / (psi_soil - psi_collar)",
                            references="Couvreur et al. 2012; Meunier et al. 2020"))
    potential: float = state_variable(unit="hPa", description="", default=0., location="scalar")


def test_an_upscaled_variable_declares_what_it_is_the_equivalent_of():
    table = RootSystem.declarations().set_index("name")
    declared = table.loc["root_system_conductance", "upscaling"]
    assert declared.of == ("wall_conductance", "membrane_conductance") and declared.law == "network"
    assert str(declared).startswith("network of wall_conductance, membrane_conductance: uniform soil")
    assert table.loc["potential", "upscaling"] is None
    with pytest.raises(ValueError, match="law"):
        upscaling(of="k", law="harmonic")
    with pytest.raises(ValueError, match="of="):
        upscaling(of=())


def test_the_frames_carry_the_upscaling():
    from openalea.metafspm.data_structure.data_api import ArrayDataStructure
    from openalea.metafspm.io.encoding3d import to_frame
    grid = ArrayDataStructure(shape=(1,))
    component = RootSystem(data_structure=grid)
    frame = to_frame({"roots": grid}, components={"roots": [component]})
    variable = frame.data_structures[0].scalars.variables["root_system_conductance"]
    assert "[upscaling: network of wall_conductance, membrane_conductance" in variable.description
