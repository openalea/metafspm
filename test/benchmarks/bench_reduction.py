"""
The reduction's benchmark (devplan_release_optimisation.md §5): solve.reduction on a toy plant of instanced sections
of a few hundred cells (a chain of cortical cells and their walls, two xylem vessels: the ports, three soil walls:
kept), standalone (no port). Not part of the test suite: it asserts nothing and prints a table.

    python test/benchmarks/bench_reduction.py [--cells 200] [--kinds 8] [--groups 1000,10000,80000] [--check]

- the time per kind evaluation (Reduction.group_kinds on --kinds groups of distinct kinds: their soil potentials
  differ, so each is its own kind);
- Reduction.terms() on a plant of 1k, 10k, 80k groups of one kind (the kind evaluated before the timed call; the
  first call builds the groups' table and the terms' sparse pattern, the next ones fill its values), and
  Reduction.assignment (the groups' kinds, group_kinds' arrays): linear in the groups (O-5), --check asserts it
  (10× the groups within 15× the time).
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "helpers"))

from openalea.metafspm.coupling.component import FunctionalComponent, StructuralComponent, parameter, state_variable  # noqa: E402
from openalea.metafspm.coupling.composite_wrapper import CompositeModel                     # noqa: E402
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales    # noqa: E402
from openalea.metafspm.data_structure.mpg import MPG                                        # noqa: E402
from openalea.metafspm.scene.population import Planting, planting, population_data_structure  # noqa: E402
from openalea.metafspm.solve.decorator import boundary_set, graph_jacobian, graph_system, node_balance  # noqa: E402
from openalea.metafspm.solve.reduction import Reduction                                     # noqa: E402

from growth import DOC                                                                      # noqa: E402

PROTOPLAST, SOIL_WALL, WALL = 0., 1., 3.
SETTINGS = {"cells": 200, "groups": 4}


def section_template(cells: int):
    """A section of *cells* cortical cells in a chain (each its protoplast and its wall, the walls in a chain, the
    protoplasts joined by plasmodesmata), the first three walls on the soil, two xylem vessels on the last walls."""
    from openalea.metafspm.data_structure.instances import SectionTemplate
    g = MPG()
    s, L = g.scales, g.labels
    segment = g._add_system_root_at_scale(s.SubOrgan, label=L.SubOrgan.RootSegment)
    first = g._id + 1

    def props(scale, **values):
        out = dict(PropsConfig(scale=scale, edge_type='/'))
        out.update(values)
        return out

    cortex, xylem = g.add_components_bulk(segment, 2, **props(s.Layer, label=[L.Layer.CorticalParenchyma,
                                                                              L.Layer.MetaXylem]))
    cortical = g.add_components_bulk(cortex, cells, **props(s.Cell, label=L.Cell.CorticalParenchyma))
    vessels = g.add_components_bulk(xylem, 2, **props(s.Cell, label=L.Cell.MetaXylem, vessel_index=[0., 1.]))
    protoplasts = g.add_components_bulk(cortical, cells, **props(s.Compartment, label=L.Compartment.Symplastic,
                                                                 kind=PROTOPLAST))
    protoplasts += g.add_components_bulk(vessels, 2, **props(s.Compartment, label=L.Cell.MetaXylem,
                                                             vessel_index=[0., 1.], kind=PROTOPLAST))
    walls = g.add_components_bulk(s.anchors[s.Compartment], cells, topo_parents=[segment] * cells,
                                  **props(s.Compartment, label=L.Compartment.Apoplastic,
                                          kind=[SOIL_WALL] * 3 + [WALL] * (cells - 3)))
    links = [(protoplasts[i], walls[i], L.Connection.Transmembrane) for i in range(cells)]
    links += [(protoplasts[i], protoplasts[i + 1], L.Connection.Symplastic) for i in range(cells - 1)]
    links += [(walls[i], walls[i + 1], L.Connection.Apoplastic) for i in range(cells - 1)]
    links += [(protoplasts[cells], walls[-1], L.Connection.Transmembrane),
              (protoplasts[cells + 1], walls[-2], L.Connection.Transmembrane)]
    g.add_components_bulk(s.anchors[s.Connection], len(links), topo_parents=[segment] * len(links),
                          **props(s.Connection, n_id_a=[a for a, _, _ in links], n_id_b=[b for _, b, _ in links],
                                  label=[label for _, _, label in links]))
    g.convert_properties_to_arraydict()
    return SectionTemplate.capture(g, segment, range(first, g._id + 1), key=("bench section", cells))


_TEMPLATES = {}


def template(cells: int):
    if cells not in _TEMPLATES:
        _TEMPLATES[cells] = section_template(cells)
    return _TEMPLATES[cells]


class BenchSegments(StructuralComponent):
    """SETTINGS["groups"] root segments in a chain under each plant, each an instance of the bench section."""

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        s, L = g.scales, g.labels
        parent = plant
        for scale, label in ((s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root), (s.Phytomer, L.Phytomer.Root),
                             (s.Organ, L.Organ.RootInternode)):
            parent = g.add_component(parent, **PropsConfig(scale=scale, edge_type='/', label=label))
        section = template(SETTINGS["cells"])
        reduction = getattr(g, "reduction", None)
        previous = None
        for _ in range(SETTINGS["groups"]):
            props = PropsConfig(scale=s.SubOrgan, edge_type='<' if previous else '/', label=L.SubOrgan.RootSegment)
            previous = g.add_child(previous, **props) if previous else g.add_component(parent, **props)
            reduced = reduction is not None and reduction.reduces(g, previous)
            g.add_instance(previous, reduction.template_for(section) if reduced else section)


def _collar(ds):
    return (ds.get("label") == ds.mtg.labels.Cell.MetaXylem) & (ds.owner("SubOrgan") == 0)


@dataclass
class BenchHydraulics(FunctionalComponent):
    """Steady water potential: the soil walls joined to the soil, the first section's xylem held."""
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


def _xylem(section):
    rows, values = section.properties["label"]
    out = np.zeros(section.size, dtype=bool)
    out[rows] = np.asarray(values, dtype=float) == MPG().labels.Cell.MetaXylem
    return out


def _soil_walls(section):
    rows, values = section.properties["kind"]
    out = np.zeros(section.size, dtype=bool)
    out[rows] = np.asarray(values, dtype=float) == SOIL_WALL
    return out


def _wiring(g):
    return [dict(node_label=g.labels.Cell.MetaXylem, edge_label=g.labels.Connection.Apoplastic,
                 ordering="vessel_index", match="equal")]


class BenchPlant(CompositeModel):
    from_scale, nodes, initiators = "SubOrgan", "Compartment", (BenchSegments,)
    reduction = Reduction(location="SubOrgan", where={"label": "RootSegment"}, ports=_xylem, keep=_soil_walls,
                          kind={"soil_water_potential": None}, systems=("BenchHydraulics._hydraulics",))
    wiring = staticmethod(_wiring)

    parts = {"hydraulics": BenchHydraulics}


def plant(groups: int, cells: int):
    """A bench plant of *groups* reduced sections of *cells* cells: (its population, its reduction)."""
    SETTINGS.update(groups=groups, cells=cells)
    population = population_data_structure(planting([Planting()], BenchPlant, {}), BenchPlant)
    return population, population.data_structure.structure.reduction


def kind_evaluation(kinds: int, cells: int) -> tuple:
    """(seconds per kind, interior cells per kind): *kinds* groups of distinct soil potentials, evaluated at once."""
    population, reduction = plant(kinds, cells)
    ds = population.data_structure
    owner = np.asarray(ds.owner("SubOrgan"))
    ds.set("soil_water_potential", -0.1 - 0.01 * owner.astype(float))
    groups = reduction.reduced_groups(ds)
    started = time.perf_counter()
    found = reduction.group_kinds(ds, groups)
    seconds = time.perf_counter() - started
    kind = next(iter(found[0].values()))
    return seconds / kinds, len(kind.interior_rows)


def terms(groups: int, cells: int) -> tuple:
    """(seconds of one terms() call, seconds of its first call, which evaluates the kind, seconds of one
    assignment) on *groups* groups of one kind."""
    population, reduction = plant(groups, cells)
    ds = population.data_structure
    component, system = population.instance.hydraulics, "BenchHydraulics._hydraulics"
    nodes = ds.entity_ids("node")
    started = time.perf_counter()
    reduction.terms(component, system, ds, nodes)
    first = time.perf_counter() - started
    started = time.perf_counter()
    reduction.terms(component, system, ds, nodes)
    seconds = time.perf_counter() - started
    started = time.perf_counter()
    reduction.assignment(ds)
    return seconds, first, time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--cells", type=int, default=200)
    parser.add_argument("--kinds", type=int, default=8)
    parser.add_argument("--groups", default="1000,10000,80000")
    parser.add_argument("--check", action="store_true", help="assert terms() and assignment linear in the groups")
    arguments = parser.parse_args(argv)
    per_kind, interior = kind_evaluation(arguments.kinds, arguments.cells)
    print(f"{'measure':<40} {'value':>12}")
    print(f"{f'kind evaluation ({interior} interior nodes)':<40} {per_kind:>10.3f} s")
    measured = {}
    for groups in [int(value) for value in arguments.groups.split(",") if value]:
        seconds, first, assigned = terms(groups, arguments.cells)
        measured[groups] = (seconds, assigned)
        print(f"{f'terms() at {groups} groups':<40} {seconds:>10.3f} s   (first call {first:.3f} s, "
              f"assignment {assigned:.3f} s)")
    if arguments.check:
        sizes = sorted(measured)
        for small, large in zip(sizes, sizes[1:]):
            factor = 1.5 * large / small
            for position, name in ((0, "terms()"), (1, "assignment")):
                ratio = measured[large][position] / max(measured[small][position], 1e-4)
                assert ratio <= factor, f"{name}: {large} groups {ratio:.1f}× the time of {small} (at most {factor:g}×)"
        print("linear in the groups: checked")


if __name__ == "__main__":
    main()
