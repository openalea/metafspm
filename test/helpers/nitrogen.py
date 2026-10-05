"""
UC1 components shared by the tests: nitrogen transport in the xylem, a concentration on the nodes and an axial flux
on the edges, with the balance and the law in their residual, explicit and rate forms, on the simple seedling's
graph at a given scale.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import boundary_condition, edge_law, graph_output, graph_system, node_balance
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=-1e4, max_value=1e4, value_comment="", references="",
           DOI=[])
DT, K = 0.5, 0.07
SCALES = {"SubOrgan": (14, 13), "Organ": (8, 7)}


def nitrogen_ds(scale="SubOrgan"):
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(getattr(g.scales, scale))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=getattr(g.scales, scale))
    is_root = np.zeros(ds.n_nodes())
    is_root[ds.roots()[0]] = 1.
    ds.register("is_root", is_root, location="node")
    return ds


def incidence(ds):
    return ds.incidence_matrix().toarray()


# ---------------------------------------------------------------- the equations

def balance(self, concentration, axial_flux, radial_solute_input):
    B = self._graph_view.incidence
    return ((concentration - self.previous("concentration")) / self.dt
            + np.asarray(B @ axial_flux).reshape(-1) - radial_solute_input)


def balance_target(self, axial_flux, radial_solute_input):
    """explicit=True: the new concentration, c_old - dt (B q - J)."""
    B = self._graph_view.incidence
    return self.previous("concentration") - self.dt * (np.asarray(B @ axial_flux).reshape(-1) - radial_solute_input)


def concentration_rate(self, axial_flux, radial_solute_input):
    """The rate form: dc/dt = J - B q."""
    B = self._graph_view.incidence
    return radial_solute_input - np.asarray(B @ axial_flux).reshape(-1)


def fick(self, concentration, axial_flux, K_axial):
    B = self._graph_view.incidence
    return axial_flux - K_axial * np.asarray(B.T @ concentration).reshape(-1)


def fick_target(self, concentration, K_axial):
    """explicit=True: the flux itself, K (B^T c)."""
    B = self._graph_view.incidence
    return K_axial * np.asarray(B.T @ concentration).reshape(-1)


# ---------------------------------------------------------------- the components, one graph system each

@dataclass
class NitrogenFields(FunctionalComponent):
    concentration: float = state_variable(**DOC, initialize=0.5, location="node", state_variable_type="intensive")
    axial_flux: float = state_variable(**DOC, initialize=0., location="edge", state_variable_type="extensive")
    K_axial: float = parameter(**DOC, by="NitrogenFields", default=0.05, location="edge")
    radial_solute_input: float = input_variable(**DOC, by="RadialExchange", initialize=0., location="node")
    c_dirichlet: float = parameter(**DOC, by="NitrogenFields", default=0., location="node")
    q_boundary: float = parameter(**DOC, by="NitrogenFields", default=0., location="node")
    time_step = DT


@dataclass
class Transport(NitrogenFields):
    @graph_system(node_unknowns=["concentration"], edge_unknowns=["axial_flux"], schedule_as="state")
    class _solve:
        _balance = node_balance(field="concentration")(balance)
        _fick = edge_law(field="axial_flux")(fick)


@dataclass
class TransportExplicitLaw(NitrogenFields):
    @graph_system(node_unknowns=["concentration"], edge_unknowns=["axial_flux"])
    class _solve:
        _balance = node_balance(field="concentration")(balance)
        _fick = edge_law(field="axial_flux", explicit=True)(fick_target)


@dataclass
class TransportExplicitBalance(NitrogenFields):
    @graph_system(node_unknowns=["concentration"], edge_unknowns=["axial_flux"])
    class _solve:
        _balance = node_balance(field="concentration", explicit=True)(balance_target)
        _fick = edge_law(field="axial_flux")(fick)


@dataclass
class TransportWithAmount(NitrogenFields):
    @graph_system(node_unknowns=["concentration"], edge_unknowns=["axial_flux"])
    class _solve:
        _balance = node_balance(field="concentration")(balance)
        _fick = edge_law(field="axial_flux", explicit=True, integrate=True)(fick_target)


@dataclass
class TransportWithOutput(NitrogenFields):
    @graph_system(node_unknowns=["concentration"], edge_unknowns=["axial_flux"])
    class _solve:
        _balance = node_balance(field="concentration")(balance)
        _fick = edge_law(field="axial_flux")(fick)

        @graph_output("axial_divergence", location="node")
        def _divergence(self, axial_flux):
            return np.asarray(self._graph_view.incidence @ axial_flux).reshape(-1)


@dataclass
class TransportDirichlet(NitrogenFields):
    @graph_system(node_unknowns=["concentration"], edge_unknowns=["axial_flux"])
    class _solve:
        _balance = node_balance(field="concentration")(balance)
        _fick = edge_law(field="axial_flux")(fick)

        @boundary_condition("node", "dirichlet", field="concentration", select="is_root")
        def _root(self, concentration, c_dirichlet):
            return concentration - c_dirichlet


@dataclass
class TransportNeumann(NitrogenFields):
    @graph_system(node_unknowns=["concentration"], edge_unknowns=["axial_flux"])
    class _solve:
        _balance = node_balance(field="concentration", explicit=True)(balance_target)
        _fick = edge_law(field="axial_flux")(fick)

        @boundary_condition("node", "neumann", field="concentration", select="is_root")
        def _root(self, q_boundary):
            return q_boundary                                          # an inflow
