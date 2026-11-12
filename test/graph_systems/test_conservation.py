"""
Declared conservation (graph_system(conserved=...), solve.conservation) and the values before a step (data_structure.previous):
the exchanged amounts integrated over the accepted sub-steps, the change of what is held, their residual the
solver's tolerance; conservation.csv; a Summary's max_change; the wall time per component in run.json.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene.recorder import SceneRecorder, Summary
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.conservation import Conservation
from openalea.metafspm.solve.decorator import edge_law, graph_output, graph_system, node_balance

from scene_doubles import DT as SCENE_DT, RootPopulation, SceneSoil, doubles_planting, soil_translator

DOC = dict(unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="", DOI=[])
SHAPE, DX, DT = (3, 1, 4), (0.1, 0.1, 0.05), 600.
DECAY = 2e-3                                           # the bottom cells' outflow k c² V (mol s-1)


class _Equations:
    """A solute diffusing through the faces, supplied at the top, lost at the bottom by a nonlinear outflow."""

    @node_balance(field="solute")
    def _balance(self, solute, solute_flux, inflow, is_bottom, volume):
        outflow = DECAY * is_bottom * solute ** 2 * volume
        return (solute - self.previous("solute")) * volume / self.step_duration \
            + np.asarray(self.graph_view.incidence @ solute_flux).reshape(-1) - inflow + outflow

    @edge_law(field="solute_flux")
    def _fick(self, solute, solute_flux, face_area, face_distance):
        return solute_flux - 1e-6 * face_area / face_distance * np.asarray(self.graph_view.incidence.T @ solute).reshape(-1)

    @graph_output("outflow", location="node")
    def _outflow(self, solute, is_bottom, volume):
        return DECAY * is_bottom * solute ** 2 * volume


WATER = {"solute": Conservation(amount=("solute", "volume"), inflow="inflow", outflow="outflow", unit="mmol")}


@dataclass
class Soil(FunctionalComponent):
    solute: float = state_variable(unit="mol m-3", **DOC, default=0., location="cell", kind="intensive")
    solute_flux: float = state_variable(unit="mol s-1", **DOC, default=0., location="edge", kind="extensive")
    outflow: float = state_variable(unit="mol s-1", **DOC, default=0., location="cell", kind="extensive")
    inflow: float = parameter(unit="mol s-1", **DOC, by="", default=0., location="cell")
    is_bottom: float = parameter(unit="adim", **DOC, by="", default=0., location="cell")
    volume: float = parameter(unit="m3", **DOC, by="", default=0., location="cell")
    time_step = DT
    _transport = graph_system(node_unknowns=["solute"], edge_unknowns=["solute_flux"], transient=True,
                              integrate="adaptive", integration_relative_tolerance=1e-4, integration_absolute_tolerance=1e-8, absolute_tolerance=1e-14,
                              conserved=WATER)(type("_transport", (_Equations,), {}))


def _soil():
    grid = ArrayDataStructure(shape=SHAPE, spacing=DX)
    soil = Soil(data_structure=grid)
    grid.set("solute", np.linspace(1., 2., grid.n_nodes()).reshape(SHAPE))
    grid.set("volume", np.full(SHAPE, np.prod(DX)))
    grid.set("is_bottom", np.asarray(grid.layer_mask(z=-1), dtype=float).reshape(SHAPE))
    grid.set("inflow", 1e-7 * np.asarray(grid.layer_mask(z=0), dtype=float).reshape(SHAPE))
    return grid, soil


def _held(grid):
    return float(np.sum(np.asarray(grid.get("solute")) * np.asarray(grid.get("volume")))) * 1e3     # mmol


def test_the_balance_closes_to_the_solvers_tolerance_over_adaptive_sub_steps():
    """The flows of the accepted sub-steps only (an adaptive integration rejects trials and replaces its coarse
    steps): the residual is round-off, the amounts in the declared unit (mol s-1 x s -> mmol)."""
    grid, soil = _soil()
    before = _held(grid)
    for _ in range(3):
        soil()
    assert soil._last_integration["steps"] > 1                  # several sub-steps per call
    ((key, balance),) = soil.conservation().items()
    assert key == ("_transport", "solute") and balance["unit"] == "mmol"
    assert balance["change"] == pytest.approx(_held(grid) - before, rel=1e-12)
    assert balance["inflow"] == pytest.approx(3 * DT * 3 * 1e-7 * 1e3, rel=1e-12)   # 3 calls, 3 top cells
    assert balance["outflow"] > 0.
    assert abs(balance["residual"]) < 1e-9 * balance["outflow"]
    soil.conservation(reset=True)
    assert soil.conservation() == {}
    soil()
    assert soil.conservation()[key]["inflow"] == pytest.approx(DT * 3 * 1e-7 * 1e3, rel=1e-12)   # since the reset


def test_terms_whose_units_are_no_amount_of_the_quantity_raise():
    class Wrong(Soil):
        _transport = graph_system(node_unknowns=["solute"], edge_unknowns=["solute_flux"], transient=True,
                                  conserved={"solute": Conservation(amount=("solute", "volume"), inflow="volume")})(
            type("_transport", (_Equations,), {}))
    with pytest.raises(ValueError, match="no amount of"):
        Wrong(data_structure=_soil()[0])()
    with pytest.raises(ValueError, match="give what is held"):
        Conservation()
    with pytest.raises(TypeError, match="conserved="):
        graph_system(node_unknowns=["solute"], conserved={"solute": "amount"})


def test_ds_previous_is_aligned_by_id():
    """The values before the step for the entities there now: a new entity gets *missing*."""
    grid = ArrayDataStructure(shape=(4,))
    grid.register("x", np.arange(4.), location="cell")
    grid.remember_previous(["x"])
    grid.set("x", np.arange(4.) + 10.)
    np.testing.assert_array_equal(grid.get("x") - grid.previous("x"), 10.)
    with pytest.raises(KeyError, match="were not kept"):
        grid.previous("y")


# ---------------------------------------------------------------- in a Scene

def test_conservation_csv_max_change_and_component_timing(tmp_path):
    """A Scene writes each step's balances (conservation.csv), a Summary's max_change from data_structure.previous (NaN at the
    first row), the wall time per component ("<Model>.<Component>") in run.json."""

    class SoilModel(SceneSoil):
        def __init__(self, plant_data_structures, x_range, y_range, time_step):
            super().__init__(plant_data_structures, x_range, y_range, time_step)
            self.water_grid, self.water = _soil()
            self.components = [self.nitrate, self.water]

    tables = {"changes": Summary("node", max_change=["struct_mass"], population="RootPopulation"),
              "soil": Summary("cell", sum=["outflow"], max_change=["solute"], population="SoilModel_1")}
    scene = Scene(doubles_planting([RootPopulation]), environment=[SoilModel],
                  translator=soil_translator("SceneExudation"), time_step=SCENE_DT,
                  recorder=SceneRecorder(str(tmp_path / "out"), tables=tables))
    assert "SoilModel_1" in scene.data_structures()
    scene.simulate(3)
    conservation = pd.read_csv(tmp_path / "out" / "summaries" / "conservation.csv")
    assert list(conservation["time"]) == [SCENE_DT, 2 * SCENE_DT, 3 * SCENE_DT]
    assert set(conservation["component"]) == {"Soil"} and set(conservation["model"]) == {"SoilModel_1"}
    assert np.all(np.abs(conservation["residual"]) < 1e-9 * conservation["outflow"])
    changes = pd.read_csv(tmp_path / "out" / "summaries" / "hand_declared" / "changes.csv")
    assert np.isnan(changes["struct_mass_max_change"].iloc[0]) and (changes["struct_mass_max_change"].iloc[1:] > 0).all()
    soil = pd.read_csv(tmp_path / "out" / "summaries" / "hand_declared" / "soil.csv")
    assert (soil["solute_max_change"].iloc[1:] > 0).all()
    with open(tmp_path / "out" / "run.json") as file:
        timing = json.load(file)["timing"]
    assert {"RootPopulation.RootGrowthProbe", "RootPopulation.CarbonProbe", "SoilModel.Soil",
            "SoilModel.SceneSoilNitrate"} <= set(timing)
    assert timing["SoilModel.Soil"] <= timing["SoilModel"]
