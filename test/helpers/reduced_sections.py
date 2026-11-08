"""
Toy reduced sections whose membranes' conductance grows with the section's length (test_reduction_classes): a chain
of N_SEGMENTS root segments per plant, each an instance of the layered toy section (layered_anatomy), reduced when
the plant model declares a reduction; the plant models at module level (importable by name).
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

N_SEGMENTS = 6
SYSTEM = "LengthHydraulics._hydraulics"


class LengthSegments(StructuralComponent):
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
        previous = None
        for _ in range(N_SEGMENTS):
            props = PropsConfig(scale=s.SubOrgan, edge_type='<' if previous else '/', label=L.SubOrgan.RootSegment)
            previous = g.add_child(previous, **props) if previous else g.add_component(parent, **props)
            reduced = reduction is not None and reduction.reduces(g, previous)
            g.add_instance(previous, reduction.template_for(template) if reduced else template)


def _collar(ds):
    """The first section's xylem."""
    return (ds.get("label") == ds.mtg.labels.Cell.MetaXylem) & (ds.owner("SubOrgan") == 0)


def _conductances(view, K, kind, length) -> np.ndarray:
    """Each edge's conductance: K, times its section's length on the membranes and wall paths reaching a wall (a
    lateral surface), the others' K alone (a cross-section's)."""
    ends = abs(view.incidence)
    on_wall = np.asarray(ends.T @ (np.asarray(kind) == WALL).astype(float)).reshape(-1) > 0
    edge_length = 0.5 * np.asarray(ends.T @ np.asarray(length, dtype=float)).reshape(-1)
    return np.asarray(K, dtype=float) * np.where(on_wall, edge_length, 1.)


@dataclass
class LengthHydraulics(FunctionalComponent):
    """Steady water potential, the walls joined to the soil, the first section's xylem held; the lateral conductances
    proportional to the section's length."""
    water_potential: float = state_variable(**DOC, default=-0.5, location="node")
    K: float = parameter(**DOC, by="", default=1., location="edge")
    label: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
    kind: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
    length: float = parameter(**DOC, by="", default=1., location="node")
    soil_water_potential: float = parameter(**DOC, by="", default=-0.2, location="node")
    xylem_water_potential: float = parameter(**DOC, by="", default=-1., location="node")

    @graph_system(node_unknowns=["water_potential"], solver="newton")
    class _hydraulics:
        soil = boundary_set(filters={"kind": "==1"}, kind="robin", value="soil_water_potential", weight=0.6)
        collar = boundary_set(filters=_collar, kind="dirichlet", value="xylem_water_potential")

        @node_balance(field="water_potential")
        def _balance(self, water_potential, K, kind, length):
            B = self.graph_view.incidence
            conductance = _conductances(self.graph_view, K, kind, length)
            return np.asarray(B @ (conductance * np.asarray(B.T @ water_potential).reshape(-1))).reshape(-1)

        @graph_jacobian
        def _jacobian(self, K, kind, length):
            B = self.graph_view.incidence
            return (B @ diags(_conductances(self.graph_view, K, kind, length)) @ B.T).tocsr()


def _xylem(template):
    from openalea.metafspm.data_structure.mpg import MPG
    rows, values = template.properties["label"]
    out = np.zeros(template.size, dtype=bool)
    out[rows] = np.asarray(values, dtype=float) == MPG().labels.Cell.MetaXylem
    return out


def _soil_walls(template):
    rows, values = template.properties["kind"]
    out = np.zeros(template.size, dtype=bool)
    out[rows] = np.asarray(values, dtype=float) == WALL
    return out




def _length_plant(classes):
    """The toy plant, its sections reduced with kinds by length: *classes* None (exact) or a Relative; False: the
    whole network."""
    class LengthPlant(CompositeModel):
        from_scale, nodes, initiators = "SubOrgan", "Compartment", (LengthSegments,)
        reduction = None if classes is False else Reduction(
            location="SubOrgan", where={"label": "RootSegment"}, ports=_xylem, keep=_soil_walls,
            kind={"length": classes}, systems=(SYSTEM,))

        @staticmethod
        def wiring(g):
            return wiring(g)

        parts = {"hydraulics": LengthHydraulics}
    return LengthPlant


ExactLengthPlant = _length_plant(None)
ExactLengthPlant.__name__ = ExactLengthPlant.__qualname__ = "ExactLengthPlant"


def length_population(model, lengths):
    """A plant of *model* whose sections have *lengths* (one per section), not solved."""
    population = population_data_structure(planting([Planting()], model, {}), model)
    ds = population.data_structure
    owner = np.asarray(ds.owner("SubOrgan"))
    ds.set("length", np.asarray(lengths, dtype=float)[owner])
    return population
