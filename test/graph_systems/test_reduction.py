"""
solve.reduction: a graph system reduced by groups. A plant model of toy sections (test/helpers/layered_anatomy.py)
declaring a Reduction keeps its xylem (the ports, joined across sections) and its soil walls (their Robin terms,
live) and eliminates the rest, once per kind, from a library of full instances: the same potentials on the kept
nodes as the full model; the interior recovered.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags

from openalea.metafspm.coupling.component import FunctionalComponent, StructuralComponent, parameter, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.scene.population import Planting, planting, population_data_structure
from openalea.metafspm.solve.decorator import boundary_set, graph_jacobian, graph_system, node_balance
from openalea.metafspm.solve.reduction import Reduction

from growth import DOC
from layered_anatomy import WALL, section_template, wiring

N_SEGMENTS = 4


class ToySegments(StructuralComponent):
    """A chain of root segments under each plant, each an instance of the toy section (reduced when the plant model
    declares a reduction)."""

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        s, L = g.scales, g.labels
        parent = plant
        for scale, label in ((s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root), (s.Phytomer, L.Phytomer.Root),
                             (s.Organ, L.Organ.RootInternode)):
            parent = g.add_component(parent, **PropsConfig(scale=scale, edge_type='/', label=label))
        template = section_template()[0]
        reduction = getattr(g, "reduction", None)
        if reduction is not None and reduction.reduces("RootSegment"):
            template = reduction.template_for(template)
        previous = None
        for _ in range(N_SEGMENTS):
            props = PropsConfig(scale=s.SubOrgan, edge_type='<' if previous else '/', label=L.SubOrgan.RootSegment)
            previous = g.add_child(previous, **props) if previous else g.add_component(parent, **props)
            g.add_instance(previous, template)


@dataclass
class NodeHydraulics(FunctionalComponent):
    """Steady water potential: the walls joined to the soil, the first section's xylem held."""
    water_potential: float = state_variable(**DOC, default=-0.5, location="node")
    K: float = parameter(**DOC, by="", default=1., location="edge")
    label: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
    kind: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
    soil_water_potential: float = parameter(**DOC, by="", default=-0.2, location="node")
    xylem_water_potential: float = parameter(**DOC, by="", default=-1., location="node")

    @graph_system(node_unknowns=["water_potential"], solver="newton")
    class _hydraulics:
        soil = boundary_set(filters={"kind": "==1"}, kind="robin", value="soil_water_potential", weight=0.6)
        collar = boundary_set(filters=lambda ds: (ds.get("label") == ds.mtg.labels.Cell.MetaXylem)
                              & (ds.owner("SubOrgan") == 0), kind="dirichlet", value="xylem_water_potential")

        @node_balance(field="water_potential")
        def _balance(self, water_potential, K):
            B = self._graph_view.incidence
            return np.asarray(B @ (K * np.asarray(B.T @ water_potential).reshape(-1))).reshape(-1)

        @graph_jacobian
        def _jacobian(self, K):
            B = self._graph_view.incidence
            return (B @ diags(K) @ B.T).tocsr()


def _xylem(template):
    from openalea.metafspm.data_structure.mpg import MPG
    rows, values = template.properties["label"]
    out = np.zeros(template.size, dtype=bool)
    out[rows] = np.asarray(values, dtype=float) == MPG().labels.Cell.MetaXylem       # the label's code
    return out


def _soil_walls(template):
    rows, values = template.properties["kind"]
    out = np.zeros(template.size, dtype=bool)
    out[rows] = np.asarray(values, dtype=float) == WALL
    return out


def _model(reduced):
    class ToyPlant(CompositeModel):
        from_scale, nodes, initiators = "SubOrgan", "Compartment", (ToySegments,)
        reduction = Reduction(location="SubOrgan", labels=("RootSegment",), ports=_xylem, keep=_soil_walls,
                              systems=("NodeHydraulics._hydraulics",)) if reduced else None

        @staticmethod
        def wiring(g):
            return wiring(g)

        def __init__(self, data_structure, time_step):
            self.hydraulics = NodeHydraulics(data_structure=data_structure, time_step=time_step)
            self.couple(self.hydraulics)

        def run(self):
            self.hydraulics()
    return ToyPlant


def _solved(reduced):
    model = _model(reduced)
    population = population_data_structure(planting([Planting()], model, {}), model)
    population.instance.run()
    return population


def _places(ds):
    """Each node as (its section, its row in the full template): a full section's k-th node is the template's k-th
    Compartment, a reduced one's the k-th kept."""
    g = ds.structure
    template = section_template()[0]
    compartments = np.flatnonzero(template.scale == scales.Compartment)
    reduction = getattr(g, "reduction", None)
    kept = reduction._kept[reduction.template_for(template).key] if reduction is not None else compartments
    owner, out, seen = np.asarray(ds.owner("SubOrgan")), [], {}
    for o in owner.tolist():
        k = seen.get(o, 0)
        seen[o] = k + 1
        out.append((o, int(kept[k])))
    return out


def test_a_reduced_graph_system_gives_the_full_potentials_on_its_kept_nodes():
    full, reduced = _solved(False).data_structure, _solved(True).data_structure
    assert reduced.n_nodes() < full.n_nodes()
    row = {place: i for i, place in enumerate(_places(full))}
    rows = [row[place] for place in _places(reduced)]
    np.testing.assert_allclose(reduced.get("water_potential"), full.get("water_potential")[rows], rtol=1e-11)


def test_a_reduced_groups_interior_is_recovered():
    full_population, reduced_population = _solved(False), _solved(True)
    full, reduced = full_population.data_structure, reduced_population.data_structure
    g = reduced.structure
    group = 2
    kept = np.flatnonzero(np.asarray(reduced.owner("SubOrgan")) == group)
    rows, values = g.reduction.recover(reduced, group, reduced.get("water_potential")[kept])
    row = {place: i for i, place in enumerate(_places(full))}
    np.testing.assert_allclose(values, full.get("water_potential")[[row[(group, int(r))] for r in rows]], rtol=1e-11)
