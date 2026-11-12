"""
Components of the soil–plant–atmosphere water example.

    PlantWaterTransport, SoilWaterTransport
                        the steady water flow in the plants' anatomies and in the soil grid: j = k ΔΨ on the edges,
                        the balance of the nodes, the exchange with the air, and each one's other boundaries (the
                        root-soil exchange; the water table and the plants' uptake).
    SeedlingStructure, SoilStructure
                        the conductances k of the edges and the vapour conductances of the evaporating nodes, from the
                        structure: SeedlingStructure builds the plants (initiate_plant) and computes theirs,
                        SoilStructure those of the soil grid.
    The air is a constant input: its water potential at 50 % relative humidity and 20 °C, and the vapour factor at
    20 °C (AIR_WATER_POTENTIAL, VAPOUR_FACTOR).

Units: water potentials in MPa, volumes in mm3 (fluxes in mm3 s-1, conductances in mm3 s-1 MPa-1), lengths and
areas in m and m2. Volumes in mm3 keep the residuals of the solves well above the solver's absolute tolerance.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm import (boundary_condition, edge_law, FunctionalComponent, graph_output, graph_system,
                               input_variable, node_balance, parameter, postsegmentation, state_variable,
                               StructuralComponent)
from openalea.metafspm.data_structure.configs import ScalesConfig as scales

from openalea.metafspm.testing.soil import (AIR_WATER_POTENTIAL, MM3, VAPOUR_FACTOR, SoilStructure,  # noqa: F401
                                           SoilWaterTransport, air_water_potential, vapour_factor)
from seedling import AXIAL, LEAF_RADIAL, ROOT_RADIAL, STEM_RADIAL, build_seedling



# ═══════════════════════════════════════════════════════════════════════════════
# The water transport: in the plants and in the soil
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlantWaterTransport(FunctionalComponent):
    """
    The water flow in the plants, on their anatomies: the Compartments are the nodes, the Connections (radial edges
    and axial junctions) the edges. Water enters at the root epidermis from the soil and leaves at the stomatal
    cavities to the air.
    """
    water_potential: float = state_variable(unit="MPa", description="Water potential of a Compartment.",
                                            default=-0.1, scale=scales.Compartment, kind="intensive")
    water_flux: float = state_variable(unit="mm3 s-1", description="Flux through a Connection, from n_id_a to n_id_b.",
                                       default=0., scale=scales.Connection, kind="extensive")
    conductance: float = input_variable(unit="mm3 s-1 MPa-1", description="Hydraulic conductance of a Connection.",
                                        by="SeedlingStructure", default=0., scale=scales.Connection,
                                        kind="intensive")
    air_water_potential: float = parameter(unit="MPa", description="Water potential of the air (a constant input).",
                                           by="PlantWaterTransport", default=AIR_WATER_POTENTIAL)
    vapour_conductance: float = input_variable(unit="mm3 s-1 MPa-1",
                                               description="Liquid-vapour conductance of a stomatal cavity.",
                                               by="SeedlingStructure", default=0., scale=scales.Compartment,
                                               kind="intensive")
    is_evaporating: float = input_variable(unit="adim", description="1 at the stomatal cavities.", by="SeedlingStructure",
                                           default=0., scale=scales.Compartment, kind="descriptor")
    soil_water_potential: float = input_variable(unit="MPa", description="Water potential of the soil around the root.",
                                                 by="SoilWaterTransport", default=0., scale=scales.Compartment,
                                                 kind="intensive")
    soil_contact_conductance: float = input_variable(unit="mm3 s-1 MPa-1", description="Root-soil conductance.",
                                                     by="SeedlingStructure", default=0., scale=scales.Compartment,
                                                     kind="intensive")
    is_soil_contact: float = input_variable(unit="adim", description="1 at the root epidermis.", by="SeedlingStructure",
                                            default=0., scale=scales.Compartment, kind="descriptor")
    root_uptake: float = state_variable(unit="mm3 s-1", description="Water entering the root epidermis from the soil.",
                                        default=0., scale=scales.Compartment, kind="extensive")
    evaporation: float = state_variable(unit="mm3 s-1", description="Water leaving a stomatal cavity to the air.",
                                        default=0., scale=scales.Compartment, kind="extensive")

    @graph_system(node_unknowns=["water_potential"], edge_unknowns=["water_flux"], solver="newton", max_iterations=20,
                  schedule_as="state")
    class _flow:
        """Steady flow: the outflows of each Compartment balance its inflows from the soil and to the air."""

        @node_balance(field="water_potential")
        def _balance(self, water_flux):
            return np.asarray(self.graph_view.incidence @ water_flux).reshape(-1)           # the outflows

        @edge_law(field="water_flux", explicit=True)
        def _darcy(self, water_potential, conductance):
            """The flux through a Connection, j = k (Ψ_a - Ψ_b)."""
            return conductance * np.asarray(self.graph_view.incidence.T @ water_potential).reshape(-1)

        @boundary_condition("node", "neumann", field="water_potential", where={"is_evaporating": ">0"})
        def _to_the_air(self, water_potential, air_water_potential, vapour_conductance):
            """Liquid to vapour at the stomatal cavities: the inflow k_vap (Ψ_air - Ψ), negative (water leaves)."""
            return vapour_conductance * (air_water_potential - water_potential)

        @boundary_condition("node", "neumann", field="water_potential", where={"is_soil_contact": ">0"})
        def _from_the_soil(self, water_potential, soil_water_potential, soil_contact_conductance):
            """The inflow from the soil, an equation of the coupled soil water potential."""
            return soil_contact_conductance * (soil_water_potential - water_potential)

        @graph_output("root_uptake", location="node", where={"is_soil_contact": ">0"})
        def _root_uptake(self, water_potential, soil_water_potential, soil_contact_conductance):
            return soil_contact_conductance * (soil_water_potential - water_potential)

        @graph_output("evaporation", location="node", where={"is_evaporating": ">0"})
        def _evaporation(self, water_potential, air_water_potential, vapour_conductance):
            return vapour_conductance * (water_potential - air_water_potential)


# ═══════════════════════════════════════════════════════════════════════════════
# The structures: conductances from the structure
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SeedlingStructure(StructuralComponent):
    """
    The plant's hydraulic structure: initiate_plant builds each seedling (seedling.py: every scale, an anatomy per
    segment, the axial xylem junctions), and the steps compute its conductances from the structure.

    Conductances: radial edges k = k_s · L_segment (each anatomical edge, extrapolated in 3D along its segment), axial
    edges k = k_axial / L (L between the two segments' centres).

    Its named entity set (entity_sets): root_surface, the nodes in contact with the soil (is_soil_contact), which the
    plant model's mask and the soil links' mapping name.
    """
    entity_sets = {"root_surface": {"is_soil_contact": ">0"}}
    # the conductances computed from the structure
    k: float = state_variable(unit="mm3 s-1 MPa-1", description="Hydraulic conductance of a Connection.", default=0.,
                              scale=scales.Connection, kind="intensive")
    vapour_conductance: float = state_variable(unit="mm3 s-1 MPa-1",
                                               description="Liquid-vapour conductance of a stomatal cavity.",
                                               default=0., scale=scales.Compartment, kind="intensive")
    soil_contact_conductance: float = state_variable(unit="mm3 s-1 MPa-1",
                                                     description="Root-soil conductance of the root epidermis.",
                                                     default=0., scale=scales.Compartment,
                                                     kind="intensive")
    # the structure, read from the MPG (Compartment and Connection properties set by initiate_plant)
    tissue: float = state_variable(unit="adim", description="Tissue code of a Compartment.", default=0.,
                                   scale=scales.Compartment, kind="descriptor")
    organ: float = state_variable(unit="adim", description="Organ code of a Compartment (root, stem, leaf).",
                                  default=0., scale=scales.Compartment, kind="descriptor")
    is_evaporating: float = state_variable(unit="adim", description="1 at the stomatal cavities.", default=0.,
                                           scale=scales.Compartment, kind="descriptor")
    evaporating_area: float = state_variable(unit="m2", description="Leaf element area, at its stomatal cavity.",
                                             default=0., scale=scales.Compartment, kind="extensive")
    is_soil_contact: float = state_variable(unit="adim", description="1 at the root epidermis.", default=0.,
                                            scale=scales.Compartment, kind="descriptor")
    contact_length: float = state_variable(unit="m",
                                           description="Root length in contact with the soil, at its epidermis.",
                                           default=0., scale=scales.Compartment, kind="extensive")
    hydraulic_type: float = state_variable(unit="adim", description="Hydraulic type of a Connection.", default=0.,
                                           scale=scales.Connection, kind="descriptor")
    conductance_length: float = state_variable(unit="m", description="Length behind a Connection's conductance.",
                                               default=0., scale=scales.Connection, kind="extensive")
    # segment geometry, read at the segments and seen by their Compartments (for the mapping onto the soil)
    x1: float = state_variable(unit="m", description="", default=0., scale=scales.SubOrgan, location="node",
                               aggregation="broadcast", kind="descriptor")
    x2: float = state_variable(unit="m", description="", default=0., scale=scales.SubOrgan, location="node",
                               aggregation="broadcast", kind="descriptor")
    y1: float = state_variable(unit="m", description="", default=0., scale=scales.SubOrgan, location="node",
                               aggregation="broadcast", kind="descriptor")
    y2: float = state_variable(unit="m", description="", default=0., scale=scales.SubOrgan, location="node",
                               aggregation="broadcast", kind="descriptor")
    z1: float = state_variable(unit="m", description="", default=0., scale=scales.SubOrgan, location="node",
                               aggregation="broadcast", kind="descriptor")
    z2: float = state_variable(unit="m", description="", default=0., scale=scales.SubOrgan, location="node",
                               aggregation="broadcast", kind="descriptor")
    # parameters, per plant
    root_radial_k: float = parameter(unit="mm3 s-1 MPa-1 m-1",
                                     description="Root radial conductance, per edge and length.",
                                     by="SeedlingStructure", default=0.5, min_value=0.05, max_value=2.)
    stem_radial_k: float = parameter(unit="mm3 s-1 MPa-1 m-1",
                                     description="Stem radial conductance, per edge and length.",
                                     by="SeedlingStructure", default=0.1)
    leaf_radial_k: float = parameter(unit="mm3 s-1 MPa-1 m-1",
                                     description="Leaf radial conductance, per edge and length.",
                                     by="SeedlingStructure", default=1.)
    axial_k: float = parameter(unit="mm3 m s-1 MPa-1", description="Xylem axial conductivity.", by="SeedlingStructure",
                               default=0.1)
    root_soil_k: float = parameter(unit="mm3 s-1 MPa-1 m-1", description="Root-soil conductance per root length.",
                                   by="SeedlingStructure", default=0.5)
    stomatal_conductance: float = parameter(unit="mol m-2 s-1", description="Stomatal conductance to water vapour.",
                                            by="SeedlingStructure", default=0.2, min_value=0.02, max_value=0.6)
    vapour_factor: float = parameter(unit="mm3 s-1 MPa-1 per (mol m-2 s-1 m2)",
                                     description="e_sat V_w² / (R T P): from a vapour conductance to a liquid one.",
                                     by="SeedlingStructure", default=VAPOUR_FACTOR)

    @classmethod
    def initiate_plant(cls, mtg, plant, parameters):
        """Build the seedling of *plant* (its architecture, anatomies and junctions, see seedling.py)."""
        build_seedling(mtg, plant, parameters)

    # ── conductances ─────────────────────────────────────────────────────────

    @postsegmentation
    def _k(self, hydraulic_type, conductance_length, root_radial_k, stem_radial_k, leaf_radial_k, axial_k):
        radial = {ROOT_RADIAL: root_radial_k, STEM_RADIAL: stem_radial_k, LEAF_RADIAL: leaf_radial_k}
        k = np.zeros(len(hydraulic_type))
        for code, k_s in radial.items():
            chosen = hydraulic_type == code
            k[chosen] = (np.broadcast_to(k_s, k.shape) * conductance_length)[chosen]
        chosen = hydraulic_type == AXIAL
        k[chosen] = (np.broadcast_to(axial_k, k.shape) / np.maximum(conductance_length, 1e-12))[chosen]
        return k

    @postsegmentation(where={"is_soil_contact": ">0"})
    def _soil_contact_conductance(self, contact_length, root_soil_k):
        return root_soil_k * contact_length

    @postsegmentation(where={"is_evaporating": ">0"})
    def _vapour_conductance(self, evaporating_area, stomatal_conductance, vapour_factor):
        return stomatal_conductance * evaporating_area * vapour_factor


