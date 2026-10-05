"""
P7 benchmark: time per scene step by phase, for populations of 1 to 1000 plants of
about 2 000 segments, in segment and anatomy modes. Not part of the test suite.

    python test/benchmarks/bench_population.py [--plants 1,10,100,1000] [--steps 5] [--anatomy 2000,20000,200000]

The models are in-repo doubles with the cost structure of the real ones: vectorised rates and states, one graph
system (axial diffusion per plant), growth by bulk segmentation (array-style elongation, one batched vertex creation
per step, incremental update_topology), exchanges with a soil grid, and the scene recorder.
"""
import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "structure_tests"))

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import (FunctionalComponent, StructuralComponent, input_variable,
                                                   parameter, state_variable)
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import actual, edge_law, graph_system, node_balance, rate, segmentation, state

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])
DT = 3600.
SEGMENT_LENGTH = 0.01
MAIN, LATERALS, LATERAL = 1000, 5, 200          # 2 000 segments per plant


def _node(kind="descriptor", initialize=0., **options):
    return state_variable(**DOC, initialize=initialize, scale=scales.SubOrgan, state_variable_type=kind, **options)


# ---------------------------------------------------------------- plant model

@dataclass
class BenchGrowth(StructuralComponent):
    """Apices elongate (array-style); those longer than a segment are cut, all at once (bulk creation)."""
    length: float = _node(initialize=SEGMENT_LENGTH)
    is_apex: float = _node()
    x1: float = _node(on_grow="inherit")
    x2: float = _node(on_grow="inherit")
    y1: float = _node(on_grow="inherit")
    y2: float = _node(on_grow="inherit")
    z1: float = _node(on_grow="inherit")
    z2: float = _node(on_grow="inherit")
    elongation: float = parameter(**DOC, by="", default=0.3 * SEGMENT_LENGTH)

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s = g.scales
        x, y, z = (float(g.property(name)[plant]) for name in ("x", "y", "z"))
        axis = g.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=g.labels.Axis.Root))
        gu = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=g.labels.GrowthUnit.Root))
        phytomer = g.add_component(gu, **PropsConfig(scale=s.Phytomer, edge_type='/', label=g.labels.Phytomer.Root))
        organ = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=g.labels.Organ.RootInternode))
        props = PropsConfig(scale=s.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment)
        first = g.add_component(organ, **dict(props, edge_type='/'))

        def chain(parent, count):          # vids are consecutive: each new segment's parent is the previous one
            start = g._id + 1
            return g.add_components_bulk(organ, count, topo_parents=[parent] + list(range(start, start + count - 1)),
                                         **props)

        main = [first] + chain(first, MAIN - 1)
        vids, depth = list(main), list(range(MAIN))
        for k in range(LATERALS):
            base = main[(k + 1) * MAIN // (LATERALS + 1)]
            lateral = chain(base, LATERAL)
            g.property("edge_type")[lateral[0]] = '+'
            vids += lateral
            depth += [depth[main.index(base)]] * LATERAL
        tips = {main[-1]} | {vids[MAIN + (k + 1) * LATERAL - 1] for k in range(LATERALS)}
        for i, v in enumerate(vids):
            top = z - depth[i] * SEGMENT_LENGTH
            for name, value in (("length", SEGMENT_LENGTH), ("is_apex", float(v in tips)), ("x1", x), ("x2", x),
                                ("y1", y), ("y2", y), ("z1", top), ("z2", top - SEGMENT_LENGTH)):
                g.property(name)[v] = value

    @actual
    def _length(self, length, is_apex, elongation):
        return length + is_apex * elongation

    @segmentation
    def _segmentation(self):
        g, ds = self.mtg, self.data_structure
        ids = np.asarray(ds.entity_ids("node"))
        length, apex = np.asarray(ds.get("length")), np.asarray(ds.get("is_apex")) > 0
        cut = ids[apex & (length > SEGMENT_LENGTH)]
        if cut.size == 0:
            return
        props = {name: g.property(name) for name in ("length", "is_apex", "x2", "y2", "z2")}
        rest = {int(v): props["length"][v] - SEGMENT_LENGTH for v in cut}
        by_complex = {}
        for v in cut.tolist():
            by_complex.setdefault(g.complex(v), []).append(v)
        for organ, apices in by_complex.items():
            x, y, z = ([props[name][v] for v in apices] for name in ("x2", "y2", "z2"))
            g.add_components_bulk(organ, len(apices), topo_parents=apices,
                                  **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                label=g.labels.SubOrgan.RootSegment),
                                  length=[rest[v] for v in apices], is_apex=1., x1=x, x2=x, y1=y, y2=y, z1=z,
                                  z2=[zi - rest[v] for zi, v in zip(z, apices)])
            for parent in apices:
                props["length"][parent], props["is_apex"][parent] = SEGMENT_LENGTH, 0.


@dataclass
class BenchCarbon(FunctionalComponent):
    """Rhizodep-like vectorised rates and states."""
    hexose: float = _node("massic_concentration", 1.)
    struct_mass: float = _node("extensive", 1.)
    exudation: float = _node("extensive")
    consumption: float = _node("extensive")
    soil_nitrate: float = input_variable(**DOC, by="BenchSoilNitrate", initialize=0., scale=scales.SubOrgan)
    length: float = input_variable(**DOC, by="BenchGrowth", initialize=0., scale=scales.SubOrgan)
    exudation_rate: float = parameter(**DOC, by="", default=1e-6)
    consumption_rate: float = parameter(**DOC, by="", default=1e-6)

    @rate
    def _exudation(self, hexose, length, soil_nitrate, exudation_rate):
        return exudation_rate * hexose * length * (1. + soil_nitrate)

    @rate
    def _consumption(self, hexose, struct_mass, consumption_rate):
        return consumption_rate * hexose * struct_mass / (1. + hexose)

    @state
    def _hexose(self, hexose, exudation, consumption, struct_mass):
        return np.maximum(hexose - DT * (exudation + consumption) / struct_mass, 0.)


@dataclass
class BenchPhloem(FunctionalComponent):
    """One graph system: implicit axial diffusion of sucrose, solved on the whole population at once (each plant a
    connected piece of the graph)."""
    sucrose: float = _node("intensive", 1.)
    sucrose_flux: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, location="edge",
                                         mapping="child", state_variable_type="extensive")
    conductance: float = parameter(**DOC, by="", default=1e-5)
    time_step = DT

    @graph_system(node_unknowns=["sucrose"], edge_unknowns=["sucrose_flux"], solver="newton", transient=True)
    class _phloem:
        @node_balance(field="sucrose")
        def _balance(self, sucrose, sucrose_flux):
            return (sucrose - self.previous("sucrose")) / self.dt \
                + np.asarray(self._graph_view.incidence @ sucrose_flux).reshape(-1)

        @edge_law(field="sucrose_flux")
        def _law(self, sucrose, sucrose_flux, conductance):
            return sucrose_flux - conductance * np.asarray(self._graph_view.incidence.T @ sucrose).reshape(-1)


class BenchPlant:
    initiators = (BenchGrowth,)
    timings = {}

    def __init__(self, data_structure, time_step, **scenario):
        self.growth = BenchGrowth(data_structure=data_structure)
        self.carbon = BenchCarbon(data_structure=data_structure)
        self.phloem = BenchPhloem(data_structure=data_structure)
        self.components = [self.growth, self.carbon, self.phloem]

    def run(self):
        for name, component in (("rates and states", self.carbon), ("graph system", self.phloem),
                                ("growth", self.growth)):
            start = time.perf_counter()
            component()
            BenchPlant.timings[name] = BenchPlant.timings.get(name, 0.) + time.perf_counter() - start


# ---------------------------------------------------------------- environment

@dataclass
class BenchSoilNitrate(FunctionalComponent):
    exudation: float = input_variable(**DOC, by="BenchCarbon", initialize=0., location="cell",
                                      state_variable_type="extensive")
    soil_nitrate: float = state_variable(**DOC, initialize=1., location="cell", state_variable_type="intensive")

    @rate
    def _soil_nitrate(self, soil_nitrate, exudation):
        return 0.99 * soil_nitrate + exudation


class BenchSoil:
    def __init__(self, populations, scene_xrange, scene_yrange, time_step, **scenario):
        self.grid = ArrayDataStructure(shape=(20, 20, 50), dx=(scene_xrange / 20, scene_yrange / 20, 0.02))
        self.nitrate = BenchSoilNitrate(data_structure=self.grid)
        self.components = [self.nitrate]

    def run(self):
        self.nitrate()


TRANSLATOR = (Translator().link("BenchSoilNitrate", "exudation", "BenchCarbon", {"exudation": 1.})
              .link("BenchCarbon", "soil_nitrate", "BenchSoilNitrate", {"soil_nitrate": 1.}))


# ---------------------------------------------------------------- A: segment mode

def _planting(n_plants, side):
    per_row = int(np.ceil(np.sqrt(n_plants)))
    rows = [dict(plant=f"p{i}", model=BenchPlant, x=(i % per_row + 0.5) * side / per_row,
                 y=(i // per_row + 0.5) * side / per_row, z=0., rotation=0., scenario={"parameters": {}})
            for i in range(n_plants)]
    table = pd.DataFrame(rows)
    table.attrs.update(xrange=side, yrange=side)
    return table


def bench_segments(n_plants, steps, method):
    Choregrapher().reset()
    BenchPlant.timings = {}
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as folder:
        scene = Scene(_planting(n_plants, side=1.), environment=[BenchSoil], translator=TRANSLATOR, time_step=DT,
                      mapping_method=method, output_dirpath=folder, log_plants=["p0"], heavy_log_period=steps)
        built = time.perf_counter() - start
        ds = scene.populations[0].data_structure
        n0 = ds.n_nodes()
        phases = {"exchanges": 0., "soil": 0., "recorder": 0.}
        exchanges, recorder = scene.exchanges.exchange, scene.recorder.record

        def timed(name, f):
            def run(*args, **kwargs):
                t = time.perf_counter()
                f(*args, **kwargs)
                phases[name] += time.perf_counter() - t
            return run

        scene.exchanges.exchange = timed("exchanges", exchanges)
        scene.recorder.record = timed("recorder", recorder)
        scene.environment[0].run = timed("soil", scene.environment[0].run)
        scene.run()                                    # warm-up (numba, caches)
        phases = dict.fromkeys(phases, 0.)
        BenchPlant.timings = {}
        start = time.perf_counter()
        for _ in range(steps):
            scene.run()
        total = (time.perf_counter() - start) / steps
    phases.update(BenchPlant.timings)
    return dict(plants=n_plants, method=method, segments=n0, grown=ds.n_nodes() - n0, build_s=built,
                step_s=total, **{f"{k}_s": v / steps for k, v in phases.items()})


# ---------------------------------------------------------------- B: anatomy mode

def bench_anatomy(n_segments, steps):
    from anatomy import make_rooted_anatomy, wiring
    from openalea.metafspm.data_structure.data_api import MPGDataStructure
    Choregrapher().reset()
    Choregrapher().add_simulation_time_step(DT)
    start = time.perf_counter()
    g, _, _ = make_rooted_anatomy(n_segments)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
    built = time.perf_counter() - start
    carbon, phloem = AnatomyCarbon(data_structure=ds), AnatomyPhloem(data_structure=ds)
    carbon()
    phloem()
    timings = {"rates and states": 0., "graph system": 0.}
    for _ in range(steps):
        for name, component in (("rates and states", carbon), ("graph system", phloem)):
            t = time.perf_counter()
            component()
            timings[name] += time.perf_counter() - t
    return dict(segments=n_segments, compartments=ds.n_nodes(), edges=ds.n_edges(), build_s=built,
                **{f"{k}_s": v / steps for k, v in timings.items()})


@dataclass
class AnatomyCarbon(FunctionalComponent):
    hexose: float = state_variable(**DOC, initialize=1., scale=scales.Compartment, state_variable_type="intensive")
    exudation: float = state_variable(**DOC, initialize=0., scale=scales.Compartment, state_variable_type="extensive")
    exudation_rate: float = parameter(**DOC, by="", default=1e-6)

    @rate
    def _exudation(self, hexose, exudation_rate):
        return exudation_rate * hexose

    @state
    def _hexose(self, hexose, exudation):
        return np.maximum(hexose - DT * exudation, 0.)


@dataclass
class AnatomyPhloem(FunctionalComponent):
    solute: float = state_variable(**DOC, initialize=1., scale=scales.Compartment, state_variable_type="intensive")
    solute_flux: float = state_variable(**DOC, initialize=0., scale=scales.Connection, state_variable_type="extensive")
    conductance: float = parameter(**DOC, by="", default=1e-5)
    time_step = DT

    @graph_system(node_unknowns=["solute"], edge_unknowns=["solute_flux"], solver="newton", transient=True)
    class _transport:
        @node_balance(field="solute")
        def _balance(self, solute, solute_flux):
            return (solute - self.previous("solute")) / self.time_step \
                + np.asarray(self._graph_view.incidence @ solute_flux).reshape(-1)

        @edge_law(field="solute_flux")
        def _law(self, solute, solute_flux, conductance):
            return solute_flux - conductance * np.asarray(self._graph_view.incidence.T @ solute).reshape(-1)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plants", default="1,10,100,1000")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--anatomy", default="2000,20000,200000")
    parser.add_argument("--methods", default="barycentre,overlap")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    rows = [bench_segments(int(n), args.steps, method) for n in args.plants.split(",") if n
            for method in args.methods.split(",")]
    print("A. segment mode, seconds per step\n", pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
    rows = [bench_anatomy(int(n), args.steps) for n in args.anatomy.split(",") if n]
    if rows:
        print("\nB. anatomy mode, seconds per step\n", pd.DataFrame(rows).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
