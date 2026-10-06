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

from openalea.metafspm.coupling.component import (FunctionalComponent, StructuralComponent, input_variable, parameter,
                                                  state_variable)
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.solve.decorator import (boundary_condition, edge_law, graph_output, graph_system,
                                               node_balance, postsegmentation)
from openalea.metafspm.solve.solver import NewtonSolver

from seedling import AXIAL, LEAF_RADIAL, ROOT_RADIAL, STEM_RADIAL, build_seedling

R, WATER_MOLAR_VOLUME, PRESSURE = 8.314, 1.8e-5, 101325.          # J mol-1 K-1, m3 mol-1, Pa
MM3 = 1e9                                                           # mm3 per m3


def saturated_vapour_pressure(temperature):
    """Pa, at temperature (°C) (Tetens)."""
    return 610.78 * np.exp(17.27 * temperature / (temperature + 237.3))


def air_water_potential(relative_humidity, temperature):
    """MPa: Ψ_air = (R T / V_w) ln RH."""
    return R * (temperature + 273.15) / WATER_MOLAR_VOLUME * np.log(relative_humidity) * 1e-6


def vapour_factor(temperature):
    """
    e_sat V_w² / (R T P), in mm3 s-1 MPa-1 per (mol m-2 s-1 m2): it turns a vapour conductance (mol m-2 s-1) times a
    surface into a liquid conductance, the vapour flux driven by the vapour pressure difference linearised in Ψ.
    """
    kelvin = temperature + 273.15
    return saturated_vapour_pressure(temperature) * WATER_MOLAR_VOLUME ** 2 / (R * kelvin * PRESSURE) * 1e6 * MM3


AIR_WATER_POTENTIAL = float(air_water_potential(0.5, 20.))          # about -93.9 MPa: a dry, realistic air
VAPOUR_FACTOR = float(vapour_factor(20.))


def _doc(unit, description, lower=-1e9, upper=1e9):
    return dict(unit=unit, unit_comment="", description=description, min_value=lower, max_value=upper,
                value_comment="", references="", DOI=[])


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
    water_potential: float = state_variable(**_doc("MPa", "Water potential of a Compartment."), initialize=-0.1,
                                            scale=scales.Compartment, state_variable_type="intensive")
    water_flux: float = state_variable(**_doc("mm3 s-1", "Flux through a Connection, from n_id_a to n_id_b."),
                                       initialize=0., scale=scales.Connection, state_variable_type="extensive")
    conductance: float = input_variable(**_doc("mm3 s-1 MPa-1", "Hydraulic conductance of a Connection."),
                                        by="SeedlingStructure", initialize=0., scale=scales.Connection,
                                        state_variable_type="intensive")
    air_water_potential: float = parameter(**_doc("MPa", "Water potential of the air (a constant input)."),
                                           by="PlantWaterTransport", default=AIR_WATER_POTENTIAL)
    vapour_conductance: float = input_variable(**_doc("mm3 s-1 MPa-1", "Liquid-vapour conductance of a stomatal "
                                                      "cavity."), by="SeedlingStructure", initialize=0.,
                                               scale=scales.Compartment, state_variable_type="intensive")
    is_evaporating: float = input_variable(**_doc("-", "1 at the stomatal cavities."), by="SeedlingStructure",
                                           initialize=0., scale=scales.Compartment, state_variable_type="descriptor")
    soil_water_potential: float = input_variable(**_doc("MPa", "Water potential of the soil around the root."),
                                                 by="SoilWaterTransport", initialize=0., scale=scales.Compartment,
                                                 state_variable_type="intensive")
    soil_contact_conductance: float = input_variable(**_doc("mm3 s-1 MPa-1", "Root-soil conductance."),
                                                     by="SeedlingStructure", initialize=0., scale=scales.Compartment,
                                                     state_variable_type="intensive")
    is_soil_contact: float = input_variable(**_doc("-", "1 at the root epidermis."), by="SeedlingStructure",
                                            initialize=0., scale=scales.Compartment, state_variable_type="descriptor")
    root_uptake: float = state_variable(**_doc("mm3 s-1", "Water entering the root epidermis from the soil."),
                                        initialize=0., scale=scales.Compartment, state_variable_type="extensive")
    evaporation: float = state_variable(**_doc("mm3 s-1", "Water leaving a stomatal cavity to the air."),
                                        initialize=0., scale=scales.Compartment, state_variable_type="extensive")

    @graph_system(node_unknowns=["water_potential"], edge_unknowns=["water_flux"], solver=NewtonSolver, max_iter=20,
                  schedule_as="state")
    class _flow:
        """Steady flow: the outflows of each Compartment balance its inflows from the soil and to the air."""

        @node_balance(field="water_potential")
        def _balance(self, water_flux):
            return np.asarray(self._graph_view.incidence @ water_flux).reshape(-1)           # the outflows

        @edge_law(field="water_flux", explicit=True)
        def _darcy(self, water_potential, conductance):
            """The flux through a Connection, j = k (Ψ_a - Ψ_b)."""
            return conductance * np.asarray(self._graph_view.incidence.T @ water_potential).reshape(-1)

        @boundary_condition("node", "neumann", field="water_potential", filters={"is_evaporating": ">0"})
        def _to_the_air(self, water_potential, air_water_potential, vapour_conductance):
            """Liquid to vapour at the stomatal cavities: the inflow k_vap (Ψ_air - Ψ), negative (water leaves)."""
            return vapour_conductance * (air_water_potential - water_potential)

        @boundary_condition("node", "neumann", field="water_potential", filters={"is_soil_contact": ">0"})
        def _from_the_soil(self, water_potential, soil_water_potential, soil_contact_conductance):
            """The inflow from the soil, an equation of the coupled soil water potential."""
            return soil_contact_conductance * (soil_water_potential - water_potential)

        @graph_output("root_uptake", location="node", filters={"is_soil_contact": ">0"})
        def _root_uptake(self, water_potential, soil_water_potential, soil_contact_conductance):
            return soil_contact_conductance * (soil_water_potential - water_potential)

        @graph_output("evaporation", location="node", filters={"is_evaporating": ">0"})
        def _evaporation(self, water_potential, air_water_potential, vapour_conductance):
            return vapour_conductance * (water_potential - air_water_potential)


@dataclass
class SoilWaterTransport(FunctionalComponent):
    """
    The water flow in the soil grid: the cells are the nodes, the faces between them the edges. Water comes from
    the water table, through the grid's bottom face, leaves at the surface to the air, and is taken up by the
    plants' roots.
    """
    water_potential: float = state_variable(**_doc("MPa", "Water potential of a cell."), initialize=-0.1,
                                            location="cell", state_variable_type="intensive")
    water_flux: float = state_variable(**_doc("mm3 s-1", "Flux through a face, towards increasing coordinates."),
                                       initialize=0., location="edge", state_variable_type="extensive")
    conductance: float = input_variable(**_doc("mm3 s-1 MPa-1", "Hydraulic conductance of a face."),
                                        by="SoilStructure", initialize=0., location="edge",
                                        state_variable_type="intensive")
    air_water_potential: float = parameter(**_doc("MPa", "Water potential of the air (a constant input)."),
                                           by="SoilWaterTransport", default=AIR_WATER_POTENTIAL, location="cell")
    vapour_conductance: float = input_variable(**_doc("mm3 s-1 MPa-1", "Liquid-vapour conductance of a surface "
                                                      "cell."), by="SoilStructure", initialize=0., location="cell",
                                               state_variable_type="intensive")
    is_evaporating: float = input_variable(**_doc("-", "1 at the surface."), by="SoilStructure", initialize=0.,
                                           location="cell", state_variable_type="descriptor")
    is_water_table: float = input_variable(**_doc("-", "1 at the water table."), by="SoilStructure", initialize=0.,
                                           location="cell", state_variable_type="descriptor")
    water_table_potential: float = parameter(**_doc("MPa", "Water potential at the water table."),
                                             by="SoilWaterTransport", default=0., location="cell")
    water_table_conductance: float = input_variable(**_doc("mm3 s-1 MPa-1", "Conductance between a bottom cell's "
                                                           "centre and the water table."), by="SoilStructure",
                                                    initialize=0., location="cell", state_variable_type="intensive")
    plant_uptake: float = input_variable(**_doc("mm3 s-1", "Water taken up by the plants from a cell."),
                                         by="PlantWaterTransport", initialize=0., location="cell",
                                         state_variable_type="extensive")
    evaporation: float = state_variable(**_doc("mm3 s-1", "Water leaving a surface cell to the air."),
                                        initialize=0., location="cell", state_variable_type="extensive")

    @graph_system(node_unknowns=["water_potential"], edge_unknowns=["water_flux"], solver=NewtonSolver, max_iter=20,
                  schedule_as="state")
    class _flow:
        """Steady flow: the outflows of each cell balance the water table, the plants' uptake and the evaporation."""

        @node_balance(field="water_potential")
        def _balance(self, water_flux):
            return np.asarray(self._graph_view.incidence @ water_flux).reshape(-1)           # the outflows

        @edge_law(field="water_flux", explicit=True)
        def _darcy(self, water_potential, conductance):
            """
            The flux through a face, j = k (Ψ_low - Ψ_high). The grid's edge difference is consistent between cells of
            different sizes (adaptive grids), and the two-point difference on regular ones.
            """
            return conductance * np.asarray(self._graph_view.edge_difference @ water_potential).reshape(-1)

        @boundary_condition("node", "neumann", field="water_potential", filters={"is_water_table": ">0"})
        def _water_table(self, water_potential, water_table_potential, water_table_conductance):
            """
            The water table at the bottom face of the bottom cells, held at its potential: the inflow through that
            face, k_table (Ψ_table - Ψ), k_table = K A / (h/2). Its depth is the grid's, whatever the cells' size.
            """
            return water_table_conductance * (water_table_potential - water_potential)

        @boundary_condition("node", "neumann", field="water_potential", filters={"is_evaporating": ">0"})
        def _to_the_air(self, water_potential, air_water_potential, vapour_conductance):
            """Liquid to vapour at the surface: the inflow k_vap (Ψ_air - Ψ), negative (water leaves)."""
            return vapour_conductance * (air_water_potential - water_potential)

        @boundary_condition("node", "neumann", field="water_potential")
        def _taken_by_the_plants(self, plant_uptake):
            return -plant_uptake

        @graph_output("evaporation", location="node", filters={"is_evaporating": ">0"})
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
    """
    # the conductances computed from the structure
    k: float = state_variable(**_doc("mm3 s-1 MPa-1", "Hydraulic conductance of a Connection."), initialize=0.,
                              scale=scales.Connection, state_variable_type="intensive")
    vapour_conductance: float = state_variable(**_doc("mm3 s-1 MPa-1", "Liquid-vapour conductance of a stomatal "
                                                      "cavity."), initialize=0., scale=scales.Compartment,
                                               state_variable_type="intensive")
    soil_contact_conductance: float = state_variable(**_doc("mm3 s-1 MPa-1", "Root-soil conductance of the root "
                                                            "epidermis."), initialize=0., scale=scales.Compartment,
                                                     state_variable_type="intensive")
    # the structure, read from the MPG (Compartment and Connection properties set by initiate_plant)
    tissue: float = state_variable(**_doc("-", "Tissue code of a Compartment."), initialize=0.,
                                   scale=scales.Compartment, state_variable_type="descriptor")
    organ: float = state_variable(**_doc("-", "Organ code of a Compartment (root, stem, leaf)."), initialize=0.,
                                  scale=scales.Compartment, state_variable_type="descriptor")
    is_evaporating: float = state_variable(**_doc("-", "1 at the stomatal cavities."), initialize=0.,
                                           scale=scales.Compartment, state_variable_type="descriptor")
    evaporating_area: float = state_variable(**_doc("m2", "Leaf element area, at its stomatal cavity."),
                                             initialize=0., scale=scales.Compartment, state_variable_type="extensive")
    is_soil_contact: float = state_variable(**_doc("-", "1 at the root epidermis."), initialize=0.,
                                            scale=scales.Compartment, state_variable_type="descriptor")
    contact_length: float = state_variable(**_doc("m", "Root length in contact with the soil, at its epidermis."),
                                           initialize=0., scale=scales.Compartment, state_variable_type="extensive")
    hydraulic_type: float = state_variable(**_doc("-", "Hydraulic type of a Connection."), initialize=0.,
                                           scale=scales.Connection, state_variable_type="descriptor")
    conductance_length: float = state_variable(**_doc("m", "Length behind a Connection's conductance."),
                                               initialize=0., scale=scales.Connection, state_variable_type="extensive")
    # segment geometry, read at the segments and seen by their Compartments (for the mapping onto the soil)
    x1: float = state_variable(**_doc("m", ""), initialize=0., scale=scales.SubOrgan, location="node",
                               mapping="broadcast", state_variable_type="descriptor")
    x2: float = state_variable(**_doc("m", ""), initialize=0., scale=scales.SubOrgan, location="node",
                               mapping="broadcast", state_variable_type="descriptor")
    y1: float = state_variable(**_doc("m", ""), initialize=0., scale=scales.SubOrgan, location="node",
                               mapping="broadcast", state_variable_type="descriptor")
    y2: float = state_variable(**_doc("m", ""), initialize=0., scale=scales.SubOrgan, location="node",
                               mapping="broadcast", state_variable_type="descriptor")
    z1: float = state_variable(**_doc("m", ""), initialize=0., scale=scales.SubOrgan, location="node",
                               mapping="broadcast", state_variable_type="descriptor")
    z2: float = state_variable(**_doc("m", ""), initialize=0., scale=scales.SubOrgan, location="node",
                               mapping="broadcast", state_variable_type="descriptor")
    # parameters, per plant
    root_radial_k: float = parameter(**_doc("mm3 s-1 MPa-1 m-1", "Root radial conductance, per edge and length."),
                                     by="SeedlingStructure", default=0.5)
    stem_radial_k: float = parameter(**_doc("mm3 s-1 MPa-1 m-1", "Stem radial conductance, per edge and length."),
                                     by="SeedlingStructure", default=0.1)
    leaf_radial_k: float = parameter(**_doc("mm3 s-1 MPa-1 m-1", "Leaf radial conductance, per edge and length."),
                                     by="SeedlingStructure", default=1.)
    axial_k: float = parameter(**_doc("mm3 m s-1 MPa-1", "Xylem axial conductivity."), by="SeedlingStructure",
                               default=0.1)
    root_soil_k: float = parameter(**_doc("mm3 s-1 MPa-1 m-1", "Root-soil conductance per root length."),
                                   by="SeedlingStructure", default=0.5)
    stomatal_conductance: float = parameter(**_doc("mol m-2 s-1", "Stomatal conductance to water vapour."),
                                            by="SeedlingStructure", default=0.2)
    vapour_factor: float = parameter(**_doc("mm3 s-1 MPa-1 per (mol m-2 s-1 m2)",
                                            "e_sat V_w² / (R T P): from a vapour conductance to a liquid one."),
                                     by="SeedlingStructure", default=VAPOUR_FACTOR)

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        """Build the seedling of *plant* (its architecture, anatomies and junctions, see seedling.py)."""
        build_seedling(g, plant, parameters)

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

    def __post_init__(self):
        super().__post_init__()
        # the root surface, as a mask of the DataStructure: the soil mapping maps it only (CrossMapping(mask=))
        self.data_structure.define_mask("root_surface", {"is_soil_contact": ">0"})

    @postsegmentation(filters={"is_soil_contact": ">0"})
    def _soil_contact_conductance(self, contact_length, root_soil_k):
        return root_soil_k * contact_length

    @postsegmentation(filters={"is_evaporating": ">0"})
    def _vapour_conductance(self, evaporating_area, stomatal_conductance, vapour_factor):
        return stomatal_conductance * evaporating_area * vapour_factor


@dataclass
class SoilStructure(StructuralComponent):
    """
    The soil grid's hydraulics: face conductances k = K · A / d, K the harmonic mean of the two voxels' K_sat (which
    vary between voxels, by layer); the surface layer evaporates (a soil-surface vapour conductance), the bottom
    layer's bottom face is the water table (k_table = K A / (h/2)). Regular and adaptive grids alike (faces of unequal
    cells: their own A and d).
    """
    k: float = state_variable(**_doc("mm3 s-1 MPa-1", "Hydraulic conductance of a face."), initialize=0.,
                              location="edge", state_variable_type="intensive")
    vapour_conductance: float = state_variable(**_doc("mm3 s-1 MPa-1", "Liquid-vapour conductance of a surface "
                                                      "cell."), initialize=0., location="cell",
                                               state_variable_type="intensive")
    is_evaporating: float = state_variable(**_doc("-", "1 at the surface."), initialize=0., location="cell",
                                           state_variable_type="descriptor")
    evaporating_area: float = state_variable(**_doc("m2", "Evaporating surface of a cell."), initialize=0.,
                                             location="cell", state_variable_type="extensive")
    water_table_area: float = state_variable(**_doc("m2", "Area of a bottom cell's face on the water table."),
                                             initialize=0., location="cell", state_variable_type="extensive")
    water_table_distance: float = state_variable(**_doc("m", "From a bottom cell's centre to the water table."),
                                                 initialize=1., location="cell", state_variable_type="descriptor")
    K_sat: float = parameter(**_doc("mm3 m-1 s-1 MPa-1", "Hydraulic conductivity of a voxel."), by="SoilStructure",
                             default=5., location="cell")
    soil_surface_conductance: float = parameter(**_doc("mol m-2 s-1", "Vapour conductance of the soil surface."),
                                                by="SoilStructure", default=0.005)
    vapour_factor: float = parameter(**_doc("mm3 s-1 MPa-1 per (mol m-2 s-1 m2)",
                                            "e_sat V_w² / (R T P): from a vapour conductance to a liquid one."),
                                     by="SoilStructure", default=VAPOUR_FACTOR)
    is_water_table: float = state_variable(**_doc("-", "1 at the water table."), initialize=0., location="cell",
                                           state_variable_type="descriptor")
    water_table_conductance: float = state_variable(**_doc("mm3 s-1 MPa-1", "Conductance between a bottom cell's "
                                                           "centre and the water table."), initialize=0.,
                                                    location="cell", state_variable_type="intensive")

    def __post_init__(self):
        super().__post_init__()
        self.set_layers()

    def set_layers(self):
        """
        The surface (evaporating) and bottom (water table) cells, and each surface cell's area, from the grid's
        geometry: a regular grid or an adaptive one, whose refinement calls it again (its cells change).
        """
        ds = self.data_structure
        surface = np.asarray(ds.layer_mask(z=0)).reshape(-1)                 # z points down: the surface first
        bottom = np.asarray(ds.layer_mask(z=-1)).reshape(-1)
        size = ds.cell_sizes()
        ds.set("is_evaporating", surface.astype(float))
        ds.set("is_water_table", bottom.astype(float))
        ds.set("evaporating_area", np.where(surface, size[:, 0] * size[:, 1], 0.))
        ds.set("water_table_area", np.where(bottom, size[:, 0] * size[:, 1], 0.))
        ds.set("water_table_distance", np.where(bottom, size[:, 2] / 2., 1.))

    @postsegmentation
    def _k(self, K_sat, face_area, face_distance):
        view = self.data_structure.to_graph_view()                       # faces as edges, cells as nodes
        tail, head = view.tail, view.head
        K = np.asarray(K_sat).reshape(-1)
        K_face = 2. * K[tail] * K[head] / np.maximum(K[tail] + K[head], 1e-30)
        return K_face * face_area / face_distance

    @postsegmentation(filters={"is_evaporating": ">0"})
    def _vapour_conductance(self, evaporating_area, soil_surface_conductance, vapour_factor):
        return soil_surface_conductance * evaporating_area * vapour_factor

    @postsegmentation(filters={"is_water_table": ">0"})
    def _water_table_conductance(self, K_sat, water_table_area, water_table_distance):
        """K A / (h/2): from the bottom cells' centre to their bottom face."""
        return K_sat * water_table_area / water_table_distance
