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

from openalea.metafspm.coupling.component import (FunctionalComponent, StructuralComponent, input_variable, parameter,
                                                   state_variable)
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.scene.population import Planting, planting, population_data_structure
from openalea.metafspm.solve.decorator import boundary_set, graph_jacobian, graph_system, node_balance, rate
from openalea.metafspm.solve.reduction import Reduction

from growth import DOC
from layered_anatomy import WALL, section_template, wiring

N_SEGMENTS = 4


class ToySegments(StructuralComponent):
    """A chain of root segments under each plant, each an instance of the toy section (reduced when the plant model
    declares a reduction); its sections' walls published as an entity set."""
    entity_sets = {"walls": {"kind": WALL}}

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
    """The first section's xylem (a module-level function: a checkpoint keeps the DataStructure's filters)."""
    return (ds.get("label") == ds.mtg.labels.Cell.MetaXylem) & (ds.owner("SubOrgan") == 0)


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
        collar = boundary_set(filters=_collar, kind="dirichlet", value="xylem_water_potential")

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
        reduction = Reduction(location="SubOrgan", where={"label": "RootSegment"}, ports=_xylem, keep=_soil_walls,
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


def test_the_full_template_of_a_reduced_template_is_given_by_its_key():
    """Reduction.full_template(key): the full template a reduced one came from (template_for's inverse); a key no
    reduced template has raises."""
    reduction = Reduction(location="SubOrgan", where={"label": "RootSegment"}, ports=_xylem, keep=_soil_walls,
                          systems=("NodeHydraulics._hydraulics",))
    full = section_template()[0]
    reduced = reduction.template_for(full)
    assert reduced.size < full.size
    assert reduction.full_template(reduced.key) is full
    with pytest.raises(KeyError):
        reduction.full_template(full.key)


def test_the_library_runs_at_the_scenes_time_step():
    """The library population built during a solve runs at the plants' time step (their DataStructure's), and the
    plants' step is left as it was: a daily scene's components still run one step a day."""
    model = _model(True)
    population = population_data_structure(planting([Planting()], model, {}), model, time_step=86400.)
    population.instance.run()
    assert population.data_structure.simulation_time_step == 86400.
    libraries = list(population.data_structure.structure.reduction.__dict__.get("_libraries", {}).values())
    assert all(library.population.data_structure.simulation_time_step == 86400. for library in libraries)


def test_a_reduced_plant_loaded_from_a_checkpoint_solves_as_before(tmp_path):
    """A checkpoint holds the MTG with its reduction (its templates, unbound): loaded into a freshly built
    population, the reduction is bound to that population's model and solves its groups again."""
    model = _model(True)
    solved = _solved(True).data_structure
    solved.checkpoint(str(tmp_path / "checkpoint"))
    fresh = population_data_structure(planting([Planting()], model, {}), model)
    fresh.data_structure.load_checkpoint(str(tmp_path / "checkpoint"))
    reduction = fresh.data_structure.structure.reduction
    assert reduction.instance is fresh.instance and reduction.g is fresh.data_structure.structure
    fresh.instance.run()
    np.testing.assert_allclose(fresh.data_structure.get("water_potential"), solved.get("water_potential"),
                               rtol=1e-12)


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
            return Reduction(location="SubOrgan", where={"label": "RootSegment"}, ports=_xylem, keep=_soil_walls,
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
    assert _signature(reduction, first) != _signature(reduction, second)
    assert _signature(reduction, first) == _signature(reduction, dict(first))     # by content


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


def test_the_library_runs_the_reduced_systems_components_and_their_cells_inputs_only():
    """The library runs the reduced system's component and the components giving it an input at the cells (the
    Connections' conductances here, per cell as MECHA's osmotic presets): the reduced model's potentials are the
    full model's on its kept nodes; a functional component giving it nothing at the cells (its own tables, its
    forcings, which the library has not: a shoot's development) is not run there."""
    @dataclass
    class EdgeConductance(FunctionalComponent):
        K: float = state_variable(**DOC, default=1., scale=scales.Connection, location="edge", kind="intensive")

        @rate
        def _K(self, K):
            return np.full(np.shape(K), 3.)

    @dataclass
    class GivenHydraulics(NodeHydraulics):
        K: float = input_variable(**DOC, by="EdgeConductance", default=1., scale=scales.Connection, location="edge")

    @dataclass
    class TableReader(FunctionalComponent):
        reads: float = state_variable(**DOC, default=0., scale=scales.Plant)

        @rate
        def _reads(self, reads):
            if getattr(self.data_structure.structure.reduction, "library", False):
                raise RuntimeError("TableReader run on the reduction's library: it has no tables there")
            return reads + 1.

    def with_reader(reduced):
        class ReadingPlant(_model(reduced)):
            def __init__(self, data_structure, time_step):
                self.reader = TableReader(data_structure=data_structure, time_step=time_step)
                self.conductance = EdgeConductance(data_structure=data_structure, time_step=time_step)
                self.hydraulics = GivenHydraulics(data_structure=data_structure, time_step=time_step)
                self.couple(self.reader, self.conductance, self.hydraulics)

            def run(self):
                self.reader()
                self.conductance()
                self.hydraulics()
        return ReadingPlant

    solved = {}
    for reduced in (False, True):
        model = with_reader(reduced)
        population = population_data_structure(planting([Planting()], model, {}), model)
        population.instance.run()
        solved[reduced] = population.data_structure
    full, reduced = solved[False], solved[True]
    np.testing.assert_allclose(reduced.get("reads"), 1.)
    row = {place: i for i, place in enumerate(_places(full))}
    rows = [row[place] for place in _places(reduced)]
    np.testing.assert_allclose(reduced.get("water_potential"), full.get("water_potential")[rows], rtol=1e-11)


def test_a_reduction_chosen_by_the_scenario_owns_its_keys_while_off():
    """The scenario keys a reduction function owns (its attribute scenario_keys) are the model's whether it reads
    them or not: the reduced sections given while "reduced" is off are accepted (inactive); another unknown entry
    still raises."""
    def choose(scenario):
        if not scenario.get("reduced", False):
            return None
        return Reduction(location="SubOrgan", where=scenario.get("reduced_where", {"label": "RootSegment"}),
                         ports=_xylem, keep=_soil_walls, systems=("NodeHydraulics._hydraulics",))
    choose.scenario_keys = ("reduced", "reduced_where")

    class ToyPlantOwningKeys(_model(False)):
        reduction = staticmethod(choose)

    for reduced in (False, True):
        population = population_data_structure(planting([Planting()], ToyPlantOwningKeys, {
            "reduced": reduced, "reduced_where": {"label": "RootSegment"}}), ToyPlantOwningKeys)
        assert (population.data_structure.structure.reduction is None) == (not reduced)
    with pytest.raises(ValueError, match="reduced_wher"):
        population_data_structure(planting([Planting()], ToyPlantOwningKeys, {"reduced_wher": {"label": "RootSegment"}}),
                                  ToyPlantOwningKeys)


def test_the_kinds_no_group_has_any_more_are_released():
    """A group's kind follows its kind variables (a growing section's length class, say): a solve keeps the kinds of
    the current groups only, the former ones released (their arrays freed, a cache holding one finds it empty); the
    potentials and the recovered interior are those of a plant solved with the new values from the start."""
    class KindedPlant(_model(False)):
        reduction = Reduction(location="SubOrgan", where={"label": "RootSegment"}, ports=_xylem, keep=_soil_walls,
                              kind={"soil_water_potential": None}, systems=("NodeHydraulics._hydraulics",))

    def solved(soil_water_potential):
        population = population_data_structure(planting([Planting()], KindedPlant, {}), KindedPlant)
        ds = population.data_structure
        ds.set("soil_water_potential", np.full(ds.n_nodes(), soil_water_potential))
        population.instance.run()
        return population

    population = solved(-0.2)
    ds = population.data_structure
    reduction = ds.structure.reduction
    former = [kind for by_system in reduction._kinds.values() for kind in by_system.values()]
    assert len(reduction._kinds) == 1 and former[0].D is not None
    ds.set("soil_water_potential", np.full(ds.n_nodes(), -0.3))
    population.instance.run()
    assert len(reduction._kinds) == 1 and all(kind.D is None and kind.reduction is None for kind in former)
    fresh = solved(-0.3).data_structure
    np.testing.assert_allclose(ds.get("water_potential"), fresh.get("water_potential"), rtol=1e-12)
    kept = np.flatnonzero(np.asarray(ds.owner("SubOrgan")) == 2)
    rows, values = reduction.recover(ds, 2, ds.get("water_potential")[kept])
    fresh_rows, fresh_values = fresh.structure.reduction.recover(fresh, 2, fresh.get("water_potential")[kept])
    np.testing.assert_array_equal(rows, fresh_rows)
    np.testing.assert_allclose(values, fresh_values, rtol=1e-12)


def test_a_kind_derived_value_is_computed_once_per_kind_while_its_factorisation_is_live():
    """@kind_derived(name) on the reduced system's component: run once per kind, on the library (its
    DataStructure's full group nodes given, in the kron's order), the kind's factorisation live; its value in
    kind.outputs[name]. A second solve of the same kinds runs it no more."""
    from openalea.metafspm.solve.decorator import kind_derived
    calls = []

    @dataclass
    class DerivingHydraulics(NodeHydraulics):
        @kind_derived("interior_share")
        def _interior_share(self, kind, full_template, kron, nodes):
            calls.append((kron._lu is not None, self.data_structure.structure.reduction.library, len(nodes),
                          full_template.rows_at_scale(scales.Compartment).size))
            return kron.shares(np.ones(kron.kept_nodes.size, dtype=bool), np.ones(kron.interior_nodes.size))

    class DerivingPlant(_model(False)):
        reduction = Reduction(location="SubOrgan", where={"label": "RootSegment"}, ports=_xylem, keep=_soil_walls,
                              systems=("NodeHydraulics._hydraulics",))

        def __init__(self, data_structure, time_step):
            self.hydraulics = DerivingHydraulics(data_structure=data_structure, time_step=time_step)
            self.couple(self.hydraulics)

    population = population_data_structure(planting([Planting()], DerivingPlant, {}), DerivingPlant)
    population.instance.run()
    reduction = population.data_structure.structure.reduction
    kinds = [kind for by_system in reduction._kinds.values() for kind in by_system.values()]
    assert len(calls) == len(kinds) == 1
    live, library, nodes, compartments = calls[0]
    assert live and library and nodes == compartments
    interior = kinds[0].interior_rows.size
    np.testing.assert_allclose(kinds[0].outputs["interior_share"].sum(), interior, rtol=1e-12)
    population.instance.run()
    assert len(calls) == 1


def test_entity_sets_select_by_name_on_templates_groups_and_data_structures():
    """A published entity set ("ToySegments.walls") is the rule assemblies name: on a section template (a
    reduction's ports or keep), on a DataStructure (define_mask(entities=)); where= filters the reduced groups by
    their labels and their complexes' (the label names resolved at their scale)."""
    from openalea.metafspm.coupling.declaration import entity_set, template_selection
    assert entity_set("ToySegments.walls") == {"kind": WALL}
    with pytest.raises(KeyError, match="publishes"):
        entity_set("ToySegments.conduits")
    with pytest.raises(KeyError, match="no component class"):
        entity_set("Nowhere.walls")
    template = section_template()[0]
    np.testing.assert_array_equal(template_selection("ToySegments.walls", template), _soil_walls(template))
    np.testing.assert_array_equal(template_selection(("ToySegments.walls", _xylem), template),
                                  _soil_walls(template) | _xylem(template))
    by_name = Reduction(where={"label": "RootSegment"}, ports=_xylem, keep="ToySegments.walls")
    by_rule = Reduction(where={"label": "RootSegment"}, ports=_xylem, keep=_soil_walls)
    assert by_name.template_for(template).size == by_rule.template_for(template).size < template.size
    ds = _solved(False).data_structure
    ds.define_mask("walls", entities="ToySegments.walls")
    np.testing.assert_array_equal(ds.mask("walls"), ds.get("kind") == WALL)
    with pytest.raises(TypeError):
        ds.define_mask("both", {"kind": WALL}, entities="ToySegments.walls")
    g = ds.structure
    segment = int(ds.entity_ids("SubOrgan")[0])
    assert Reduction(where={"label": "RootSegment", "Organ.label": "RootInternode"}).reduces(g, segment)
    assert not Reduction(where={"label": ["RootApex", "StemElement"]}).reduces(g, segment)
    assert not Reduction(where={"Organ.label": "Sheath"}).reduces(g, segment)
    assert Reduction().reduces(g, segment)
    with pytest.raises(KeyError, match="labels"):
        Reduction(where={"label": "Rootsegment"}).reduces(g, segment)
