"""
Scene options and population construction: the logger hook, explicit mappings (a list or a callable), environment
scenarios, the stand's size, a population in anatomy mode, plant initiators, the repartition rules given as callables,
and events across a checkpoint.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import (FunctionalComponent, StructuralComponent, input_variable,
                                                   state_variable)
from openalea.metafspm.coupling.cross import CrossMapping, LayerMapping
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene.population import build_population
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import actual, segmentation

from anatomy import add_anatomy, wiring as anatomy_wiring
from growth import DOC, RootGrowthProbe, make_chain
from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, planting, soil_translator


def _scene(**options):
    return Scene(planting([RootPopulation, Seedlings]), environment=[SceneSoil],
                 translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT, **options)


# ---------------------------------------------------------------- logger hook

class RecordingLogger:
    def __init__(self, scene, outputs_dirpath, **settings):
        self.scene, self.outputs_dirpath, self.settings = scene, outputs_dirpath, settings
        self.times, self.stopped = [], False

    def __call__(self):
        self.times.append(self.scene.time)

    def stop(self):
        self.stopped = True


def test_the_logger_is_called_after_each_step_and_stopped_at_the_end(tmp_path):
    scene = _scene(logger_class=RecordingLogger, output_dirpath=str(tmp_path), log_settings={"every": 2})
    logger = scene.logger
    assert (logger.scene, logger.outputs_dirpath, logger.settings) == (scene, str(tmp_path), {"every": 2})
    scene.simulate(3)
    assert logger.times == [DT, 2 * DT, 3 * DT]
    assert logger.stopped


# ---------------------------------------------------------------- explicit mappings

@dataclass
class ColumnHeat(FunctionalComponent):
    temperature: float = state_variable(**DOC, initialize=0., location="cell", state_variable_type="intensive")


@dataclass
class GridHeat(FunctionalComponent):
    temperature: float = input_variable(**DOC, by="ColumnHeat", initialize=-1., location="cell",
                                        state_variable_type="intensive")


class ColumnModel:
    """A 1-D column model; its DataStructure is given by its scenario, or built."""

    def __init__(self, populations, scene_xrange, scene_yrange, time_step, column=None):
        self.column = column if column is not None else ArrayDataStructure(shape=(4,), dx=0.4)
        self.heat = ColumnHeat(data_structure=self.column)
        self.column.set("temperature", [10., 12., 14., 16.])
        self.components = [self.heat]

    def run(self):
        pass


class GridModel:
    """A grid model, with the four 0.4 m layers of the column along z."""

    def __init__(self, populations, scene_xrange, scene_yrange, time_step, grid=None):
        self.scene_xrange, self.scene_yrange = scene_xrange, scene_yrange
        self.grid = grid if grid is not None else ArrayDataStructure(shape=(2, 1, 4), dx=0.4)
        self.heat = GridHeat(data_structure=self.grid)
        self.components = [self.heat]

    def run(self):
        pass


LAYERS = Translator().link("GridHeat", "temperature", "ColumnHeat", {"temperature": 1.})


def _grid_layers(grid):
    return np.asarray(grid.get("temperature")).reshape(grid.shape)[0, 0]


def test_mappings_given_as_a_list_are_used_in_the_exchanges():
    column, grid = ArrayDataStructure(shape=(4,), dx=0.4), ArrayDataStructure(shape=(2, 1, 4), dx=0.4)
    scene = Scene(planting([Seedlings]), environment=[ColumnModel, GridModel],
                  environment_scenarios=[{"column": column}, {"grid": grid}], translator=LAYERS, time_step=DT,
                  mappings=[LayerMapping(column, grid, axis="z")])
    scene.run()
    np.testing.assert_allclose(_grid_layers(grid), [10., 12., 14., 16.])


def test_mappings_given_as_a_callable_are_built_once_the_models_are():
    built = []

    def mappings(scene):
        built.append(scene)
        return [LayerMapping(scene.environment[0].column, scene.environment[1].grid, axis="z")]

    scene = Scene(planting([Seedlings]), environment=[ColumnModel, GridModel], translator=LAYERS, time_step=DT,
                  mappings=mappings)
    assert built == [scene] and isinstance(scene.mappings[0], LayerMapping)
    scene.run()
    np.testing.assert_allclose(_grid_layers(scene.environment[1].grid), [10., 12., 14., 16.])



def test_a_cross_mapping_given_replaces_the_inferred_one():
    """An explicit CrossMapping (here mapping only the first segment of each plant) replaces the inferred one."""
    def mappings(scene):
        ds = scene.populations[0].data_structure
        first = np.zeros(ds.n_nodes())
        first[ds.roots()] = 1.
        ds.register("is_first", first, location="node")
        ds.define_mask("first", {"is_first": ">0"})
        return [CrossMapping(ds, scene.environment[0].grid, mask="first")]

    scene = Scene(planting([RootPopulation]), environment=[SceneSoil],
                  translator=soil_translator("SceneExudation"), time_step=DT, mappings=mappings)
    crosses = [m for m in scene.mappings if isinstance(m, CrossMapping)]
    assert len(crosses) == 1 and crosses[0].mask == "first"
    scene.run()
    scene.run()                                                          # the soil receives the first step's exudation
    assert scene.environment[0].grid.get("exudation").sum() > 0

# ---------------------------------------------------------------- environment scenarios and the stand's size

def test_environment_scenarios_must_match_the_environment_models():
    with pytest.raises(ValueError, match="one scenario per environment model"):
        Scene(planting([Seedlings]), environment=[ColumnModel, GridModel], environment_scenarios=[{}],
              translator=LAYERS, time_step=DT)


def test_the_stand_size_defaults_to_the_planting_table_and_can_be_overridden():
    default = Scene(planting([Seedlings]), environment=[GridModel], time_step=DT)
    assert (default.scene_xrange, default.scene_yrange) == (0.8, 0.4)
    Choregrapher().reset()
    scene = Scene(planting([Seedlings]), environment=[GridModel], time_step=DT, scene_xrange=2., scene_yrange=3.)
    assert (scene.scene_xrange, scene.scene_yrange) == (2., 3.)
    assert (scene.environment[0].scene_xrange, scene.environment[0].scene_yrange) == (2., 3.)


# ---------------------------------------------------------------- initiators and anatomy mode

ORDER = []


class AnatomyInitiator(StructuralComponent):
    """Adds an anatomy to each segment of the plant built by the initiators before it."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        segments = list(g.components_at_scale(plant, g.scales.SubOrgan))
        ORDER.append(("anatomy", plant, len(segments)))
        for segment in segments:
            add_anatomy(g, segment)


class BareStructure(StructuralComponent):
    pass


def _rows(n, **parameters):
    return pd.DataFrame([dict(plant=f"p{i}", model=None, x=0., y=0., z=0., rotation=0.,
                              scenario={"parameters": dict(parameters)}) for i in range(n)])


def test_a_structural_component_without_initiate_plant_cannot_initiate_plants():
    with pytest.raises(NotImplementedError, match="BareStructure does not initiate plants"):
        build_population(_rows(1), initiators=(BareStructure,))


def test_initiators_run_in_order_each_on_every_plant():
    ORDER.clear()
    g, plants = build_population(_rows(2, n_segments=4), initiators=(RootGrowthProbe, AnatomyInitiator))
    assert ORDER == [("anatomy", plants[0], 4), ("anatomy", plants[1], 4)]     # after the roots of every plant
    anchors = set(g.scales.anchors.values())
    assert len([v for v in g.vertices(scale=g.scales.Compartment) if v not in anchors]) == 2 * 4 * 4


@dataclass
class CompartmentWater(FunctionalComponent):
    water: float = state_variable(**DOC, initialize=2., location="node", state_variable_type="intensive")


class AnatomyPopulation:
    initiators = (RootGrowthProbe, AnatomyInitiator)
    nodes = "Compartment"

    def __init__(self, data_structure, time_step):
        self.water = CompartmentWater(data_structure=data_structure)
        self.components = [self.water]

    def run(self):
        self.water()


def test_a_population_model_in_anatomy_mode_has_compartments_as_nodes():
    scene = Scene(planting([AnatomyPopulation, AnatomyPopulation]), time_step=DT)
    ds = scene.populations[0].data_structure
    assert len(ds.entity_ids("SubOrgan")) == 6                    # the segments are a coarse location
    np.testing.assert_array_equal(ds.get("water"), 2.)


def test_a_population_in_anatomy_mode_has_only_the_anatomy_compartments():
    scene = Scene(planting([AnatomyPopulation, AnatomyPopulation]), time_step=DT)
    ds = scene.populations[0].data_structure
    assert ds.n_nodes() == 2 * 3 * 4 and ds.n_edges() == 2 * 3 * 3


class WiredAnatomyPopulation(AnatomyPopulation):
    wiring = staticmethod(anatomy_wiring)


def test_a_population_model_wires_the_junctions_between_its_anatomies():
    scene = Scene(planting([WiredAnatomyPopulation, WiredAnatomyPopulation]), time_step=DT)
    ds = scene.populations[0].data_structure
    # per plant, 2 links between 3 segments: 2 xylem vessels matched by index and the 1 x 1 symplastic pair
    assert ds.n_nodes() == 2 * 3 * 4 and ds.n_edges() == 2 * 3 * 3 + 2 * 2 * 3


# ---------------------------------------------------------------- repartition rules given as callables

@dataclass
class Pools(FunctionalComponent):
    conc: float = state_variable(**DOC, initialize=3., scale=scales.SubOrgan, state_variable_type="massic_concentration")
    amount: float = state_variable(**DOC, initialize=4., scale=scales.SubOrgan, state_variable_type="extensive")


def _grown(partition_weight):
    _, ds, _ = make_chain()
    growth = RootGrowthProbe(data_structure=ds)
    growth.partition_weight = partition_weight
    Pools(data_structure=ds)
    growth()                                             # elongation, then a segmentation
    return {name: np.array(ds.get(name)) for name in ("conc", "amount", "struct_mass")}


def test_a_callable_partition_weight_is_a_variable_name_computed():
    by_name, by_callable = _grown("struct_mass"), _grown(lambda ds: np.asarray(ds.get("struct_mass")))
    for name in by_name:
        np.testing.assert_array_equal(by_callable[name], by_name[name], err_msg=name)
    assert by_name["amount"].sum() == pytest.approx(3 * 4.)            # amounts conserved, so the rule applied


@dataclass
class Branching(StructuralComponent):
    struct_mass: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="extensive")
    partition_weight = "struct_mass"
    active = staticmethod(lambda ds: np.asarray(ds.get("struct_mass")) > 0)

    @segmentation
    def _form_primordium(self):
        g = self.mtg
        if getattr(self, "primordium", None) is None:
            self.primordium = g.add_child(self.carrier, **PropsConfig(scale=g.scales.SubOrgan, edge_type='+',
                                                                      label=g.labels.SubOrgan.RootSegment,
                                                                      struct_mass=0.))

    @actual
    def _emerge(self):
        if getattr(self, "primordium", None) is not None and getattr(self, "emerge", False):
            self.mtg.property("struct_mass")[self.primordium] = 0.5


def test_an_active_rule_given_as_a_callable_defines_the_active_mask():
    _, ds, vids = make_chain()
    branching = Branching(data_structure=ds)
    branching.carrier = vids[1]
    Pools(data_structure=ds)
    branching()
    primordium = ds.index_of(branching.primordium)
    assert not ds.mask("active")[primordium]
    assert ds.get("amount")[primordium] == 0. and ds.get("conc")[primordium] == 3.
    branching.emerge = True
    branching()
    assert ds.mask("active")[primordium]
    assert ds.get("amount")[primordium] > 0.                            # split from its carrier once active


# ---------------------------------------------------------------- events across a checkpoint

def _counting_scene(fired, time):
    return Scene(planting([Seedlings]), environment=[SceneSoil], translator=soil_translator("SeedlingExudation"),
                 time_step=DT, events=[(time, lambda scene: fired.append(scene.time))])


def test_an_event_fired_before_a_checkpoint_does_not_fire_again_after_the_restore(tmp_path):
    fired = []
    scene = _counting_scene(fired, DT)
    scene.run()
    scene.run()
    assert fired == [DT]
    scene.checkpoint(str(tmp_path / "c"))
    Choregrapher().reset()
    restored = _counting_scene(fired, DT)
    restored.load_checkpoint(str(tmp_path / "c"))
    restored.run()
    assert fired == [DT]


def test_an_event_due_at_the_checkpoint_time_fires_after_the_restore(tmp_path):
    fired = []
    scene = _counting_scene(fired, 2 * DT)
    scene.run()
    scene.run()
    scene.checkpoint(str(tmp_path / "c"))
    Choregrapher().reset()
    restored = _counting_scene(fired, 2 * DT)
    restored.load_checkpoint(str(tmp_path / "c"))
    restored.run()
    assert fired == [2 * DT]


def test_an_event_between_steps_still_fires_after_a_restore(tmp_path):
    fired = []
    scene = _counting_scene(fired, 1.5 * DT)
    scene.run()
    scene.run()                                          # started at DT < 1.5 DT: not fired yet
    assert fired == []
    scene.checkpoint(str(tmp_path / "c"))
    Choregrapher().reset()
    restored = _counting_scene(fired, 1.5 * DT)
    restored.load_checkpoint(str(tmp_path / "c"))
    restored.run()                                       # starts at 2 DT >= 1.5 DT, as without the checkpoint
    assert fired == [2 * DT]
