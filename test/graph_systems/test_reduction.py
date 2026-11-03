"""
solve.reduction: a graph system reduced by groups. A plant model of toy sections (test/helpers/layered_anatomy.py)
declaring a Reduction keeps its xylem (the ports, joined across sections) and its soil walls (their Robin terms,
live) and eliminates the rest, once per kind, from a library of full instances: the same potentials on the kept
nodes as the full model; the interior recovered.
"""
from dataclasses import dataclass

import numpy as np
import pytest
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
            B = self.graph_view.incidence
            return np.asarray(B @ (K * np.asarray(B.T @ water_potential).reshape(-1))).reshape(-1)

        @graph_jacobian
        def _jacobian(self, K):
            B = self.graph_view.incidence
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


def _model_by_scenario():
    """The toy plant choosing its reduction by its scenario: "reduced" (one model class for both)."""
    class ToyPlantByScenario(_model(False)):
        @staticmethod
        def reduction(scenario):
            if not scenario.get("reduced", False):
                return None
            return Reduction(location="SubOrgan", labels=("RootSegment",), ports=_xylem, keep=_soil_walls,
                             systems=("NodeHydraulics._hydraulics",))
    return ToyPlantByScenario


def test_a_reduction_chosen_by_the_scenario():
    model = _model_by_scenario()
    solved = {}
    for reduced in (False, True):
        population = population_data_structure(planting([Planting()], model, {"reduced": reduced}), model)
        population.instance.run()
        solved[reduced] = population.data_structure
    reference = {reduced: _solved(reduced).data_structure for reduced in (False, True)}
    for reduced in (False, True):
        assert solved[reduced].n_nodes() == reference[reduced].n_nodes()
        np.testing.assert_allclose(solved[reduced].get("water_potential"), reference[reduced].get("water_potential"), rtol=1e-12)
    with pytest.raises(ValueError, match="the same for every plant"):
        population_data_structure(planting([Planting(), Planting(x=1.)], model, plant_scenarios=[
            {"reduced": True}, {"reduced": False}]), model)


def test_the_library_gets_every_per_plant_parameter_and_keys_its_kinds_by_them():
    """A plant's object parameter (e.g. a component's dict of options) reaches the reduction's library runs and tells
    kinds apart: two plants differing in it only have kinds of their own."""
    from openalea.metafspm.solve.reduction import _signature

    @dataclass
    class OptionedHydraulics(NodeHydraulics):
        options: object = parameter(**DOC, default=None, dtype="object", scale=scales.Plant)

    class OptionedPlant(_model(True)):
        def __init__(self, data_structure, time_step):
            self.hydraulics = OptionedHydraulics(data_structure=data_structure, time_step=time_step)
            self.couple(self.hydraulics)

    population = population_data_structure(planting([Planting(), Planting(x=1.)], OptionedPlant, plant_scenarios=[
        {"options": {"suberin": 1}}, {"options": {"suberin": 2}}]), OptionedPlant)
    population.instance.run()
    reduction = population.data_structure.structure.reduction
    first, second = (reduction._parameters(population.data_structure, k) for k in (0, 1))
    assert first["options"] == {"suberin": 1} and second["options"] == {"suberin": 2}
    assert _signature(first) != _signature(second)
    assert _signature(first) == _signature(dict(first))                 # by content


@pytest.mark.parametrize("n_plants", [1, 2])
def test_a_subclass_of_the_reduced_component_with_a_per_plant_parameter_gives_the_full_potentials(n_plants):
    """A reduction declares its systems by their component ("NodeHydraulics._hydraulics"): a subclass of it (here
    with a per-plant parameter more) runs the same system, reduced (its library reads its operator, not solves it),
    with the full potentials on its kept nodes."""
    @dataclass
    class FactoredHydraulics(NodeHydraulics):
        unused_factor: float = parameter(**DOC, default=1.)

    def factored(reduced):
        class FactoredPlant(_model(reduced)):
            def __init__(self, data_structure, time_step):
                self.hydraulics = FactoredHydraulics(data_structure=data_structure, time_step=time_step)
                self.couple(self.hydraulics)
        return FactoredPlant

    solved = {}
    for reduced in (False, True):
        model = factored(reduced)
        population = population_data_structure(planting([Planting(x=float(i)) for i in range(n_plants)], model,
                                                        plant_scenarios=[{}] * n_plants), model)
        population.instance.run()
        solved[reduced] = population.data_structure
    full, reduced = solved[False], solved[True]
    assert reduced.n_nodes() < full.n_nodes()
    row = {place: i for i, place in enumerate(_places(full))}
    rows = [row[place] for place in _places(reduced)]
    np.testing.assert_allclose(reduced.get("water_potential"), full.get("water_potential")[rows], rtol=1e-11)
