"""
The scenario layer: a scenario table read into {name: scenario} (values parsed, empty cells left to the defaults,
forcings from files, by model), the parameters' table, and variants within the declared bounds.
"""
from dataclasses import dataclass

import pandas as pd
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, parameter, setting
from openalea.metafspm.scene.scenarios import parameter_table, read_scenarios, variants

DOC = dict(unit="", description="")


@dataclass
class Elongation(FunctionalComponent):
    EL: float = parameter(**DOC, by="", default=1e-3, min_value=1e-4, max_value=2e-3)
    n_roots: int = parameter(**DOC, by="", default=5, min_value=1, max_value=50)
    method: str = setting(unit="", description="The solver", default="euler", choices=("euler", "rk4"))


@dataclass
class Branching(FunctionalComponent):
    IPD: float = parameter(**DOC, by="", default=6e-3)                  # no bounds
    EL: float = parameter(**DOC, by="", default=1e-3, min_value=1e-4, max_value=2e-3)


def test_a_scenario_table(tmp_path):
    (tmp_path / "rain.csv").write_text("time,rain\n0,0\n3600,2.5\n")
    (tmp_path / "scenarios.csv").write_text(
        "name,type,unit,description,reference,dry,branched\n"
        "EL,parameter,m s-1,the elongation rate,1e-3,8e-4,\n"
        "n_roots,parameter,adim,the roots,5,,12\n"
        "method,setting,,the solver,euler,rk4,\n"
        "layers,parameter,,the layers,\"[0.1, 0.2]\",,\n"
        "fixed,parameter,,a switch,True,False,None\n"
        "rain,forcing,mm,the rainfall,rain.csv,0.,\n")
    scenarios = read_scenarios(tmp_path / "scenarios.csv")
    assert list(scenarios) == ["reference", "dry", "branched"]
    rain = scenarios["reference"].pop("forcings")["rain"]
    assert scenarios["reference"] == dict(EL=1e-3, n_roots=5, method="euler", layers=[0.1, 0.2], fixed=True)
    assert rain.index.tolist() == [0., 3600.] and rain.tolist() == [0., 2.5]
    assert scenarios["dry"] == dict(EL=8e-4, method="rk4", fixed=False, forcings={"rain": 0.})
    assert scenarios["branched"] == dict(n_roots=12, fixed=None)
    assert isinstance(scenarios["branched"]["n_roots"], int)
    assert list(read_scenarios(tmp_path / "scenarios.csv", which=["dry"])) == ["dry"]
    with pytest.raises(KeyError, match="wet"):
        read_scenarios(tmp_path / "scenarios.csv", which=["wet"])


def test_a_table_by_model():
    table = pd.DataFrame({"name": ["EL", "depth", "rain"], "model": ["Roots", "Soil", None],
                          "type": ["parameter", "parameter", "forcing"], "a": [1e-3, 0.4, 2.], "b": [None, 0.6, None]})
    assert read_scenarios(table) == {"a": {"Roots": {"EL": 1e-3}, "Soil": {"depth": 0.4}, "forcings": {"rain": 2.}},
                                     "b": {"Soil": {"depth": 0.6}}}
    with pytest.raises(ValueError, match=r"\['depth'\] have no model"):
        read_scenarios(table.assign(model=["Roots", None, None]))


def test_the_parameter_table():
    table = parameter_table(Elongation, Branching, values={"IPD": 5e-3}, main=("IPD", "EL"))
    assert table["name"].tolist() == ["IPD", "EL", "n_roots", "method"]
    assert table.columns.tolist() == ["name", "value", "unit", "description", "min_value", "max_value", "component",
                                      "main"]
    row = table.set_index("name").loc["EL"]
    assert (row["min_value"], row["max_value"], row["component"]) == (1e-4, 2e-3, "Branching")   # the last one's
    assert table.set_index("name").at["IPD", "value"] == 5e-3 and table["main"].tolist() == [True, True, False, False]
    assert parameter_table(Elongation, roles=("setting",))["name"].tolist() == ["method"]
    with pytest.raises(KeyError, match="'EK'"):
        parameter_table(Elongation, values={"EK": 1.})


def test_variants():
    base = {"EL": 1e-3, "n_roots": 5}
    grid = variants(base, {"EL": (5e-4, 1.5e-3), "n_roots": [2, 8]}, levels=3, declared=[Elongation])
    assert len(grid) == 6 and grid["EL=0.0005,n_roots=8"] == {"EL": 5e-4, "n_roots": 8}
    once = variants(base, {"EL": [5e-4, 2e-3]}, design="one_at_a_time")
    assert list(once) == ["base", "EL=0.0005", "EL=0.002"] and once["base"] == base
    sampled = variants(base, {"EL": None, "n_roots": None}, design="sampling", samples=20, seed=3,
                       declared=[Elongation])
    assert list(sampled)[0] == "sample_000" and len(sampled) == 20
    values = sorted(s["EL"] for s in sampled.values())
    assert 1e-4 <= values[0] and values[-1] <= 2e-3
    bins = [int((v - 1e-4) / (1.9e-3 / 20)) for v in values]           # a Latin hypercube: one draw per stratum
    assert sorted(bins) == list(range(20))
    assert all(isinstance(s["n_roots"], int) for s in sampled.values())
    assert variants(base, {"EL": None}, design="sampling", samples=5, seed=3, declared=[Elongation]) == variants(
        base, {"EL": None}, design="sampling", samples=5, seed=3, declared=[Elongation])
    with pytest.raises(ValueError, match="outside its declared bounds"):
        variants(base, {"EL": [5e-3]}, declared=[Elongation])
    with pytest.raises(ValueError, match="no declared bounds"):
        variants(base, {"EL": None})
