"""
A soil for tests and examples: the steady water flow in a coarse fixed grid (not one of metafspm's models, a fixture).

    Soil(plant_data_structures, x_range, y_range, time_step, depth=0.4, voxel=0.05)
                        the environment model: a grid over the stand, z pointing down from the surface, 5 cm cells by
                        default (coarse: fast), its conductivity by layer (a denser subsoil), the water table at the
                        bottom, the surface evaporating
    SoilStructure, SoilWaterTransport
                        its components: the faces' conductances from the cells' K_sat, and the steady flow
                        (j = k ΔΨ on the faces, the water table, the evaporation, the plants' uptake)

The plants take up the water through a scene link: their uptake summed into the cells (plant_uptake), the soil's
water potential broadcast to their roots. Units: water potentials in MPa, volumes in mm3, lengths in m.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import (FunctionalComponent, StructuralComponent, input_variable, parameter,
                                                  state_variable)
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.solve.decorator import (boundary_condition, edge_law, graph_output, graph_system,
                                               node_balance, postsegmentation)

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


@dataclass
class SoilWaterTransport(FunctionalComponent):
    """
    The water flow in the soil grid: the cells are the nodes, the faces between them the edges. Water comes from
    the water table, through the grid's bottom face, leaves at the surface to the air, and is taken up by the
    plants' roots.
    """
    water_potential: float = state_variable(unit="MPa", description="Water potential of a cell.", default=-0.1,
                                            location="cell", kind="intensive")
    water_flux: float = state_variable(unit="mm3 s-1",
                                       description="Flux through a face, towards increasing coordinates.",
                                       default=0., location="edge", kind="extensive")
    conductance: float = input_variable(unit="mm3 s-1 MPa-1", description="Hydraulic conductance of a face.",
                                        by="SoilStructure", default=0., location="edge",
                                        kind="intensive")
    air_water_potential: float = parameter(unit="MPa", description="Water potential of the air (a constant input).",
                                           by="SoilWaterTransport", default=AIR_WATER_POTENTIAL, location="cell")
    vapour_conductance: float = input_variable(unit="mm3 s-1 MPa-1",
                                               description="Liquid-vapour conductance of a surface cell.",
                                               by="SoilStructure", default=0., location="cell",
                                               kind="intensive")
    boundary_low_z: float = input_variable(unit="adim", description="1 at the surface (the grid's top layer: z points "
                                                                    "down).", by="grid", default=0.,
                                           location="cell", kind="descriptor")
    boundary_high_z: float = input_variable(unit="adim", description="1 at the water table (the grid's bottom "
                                                                     "layer).", by="grid", default=0.,
                                            location="cell", kind="descriptor")
    water_table_potential: float = parameter(unit="MPa", description="Water potential at the water table.",
                                             by="SoilWaterTransport", default=0., location="cell")
    water_table_conductance: float = input_variable(unit="mm3 s-1 MPa-1", description="Conductance between a bottom "
                                                                                      "cell's centre and the water "
                                                                                      "table.", by="SoilStructure",
                                                    default=0., location="cell", kind="intensive")
    plant_uptake: float = input_variable(unit="mm3 s-1", description="Water taken up by the plants from a cell.",
                                         by="PlantWaterTransport", default=0., location="cell",
                                         kind="extensive")
    evaporation: float = state_variable(unit="mm3 s-1", description="Water leaving a surface cell to the air.",
                                        default=0., location="cell", kind="extensive")

    @graph_system(node_unknowns=["water_potential"], edge_unknowns=["water_flux"], solver="newton", max_iterations=20,
                  schedule_as="state")
    class _flow:
        """Steady flow: the outflows of each cell balance the water table, the plants' uptake and the evaporation."""

        @node_balance(field="water_potential")
        def _balance(self, water_flux):
            return np.asarray(self.graph_view.incidence @ water_flux).reshape(-1)           # the outflows

        @edge_law(field="water_flux", explicit=True)
        def _darcy(self, water_potential, conductance):
            """
            The flux through a face, j = k (Ψ_low - Ψ_high). The grid's edge difference is consistent between cells of
            different sizes (adaptive grids), and the two-point difference on regular ones.
            """
            return conductance * np.asarray(self.graph_view.edge_difference @ water_potential).reshape(-1)

        @boundary_condition("node", "neumann", field="water_potential", where={"boundary_high_z": ">0"})
        def _water_table(self, water_potential, water_table_potential, water_table_conductance):
            """
            The water table at the bottom face of the bottom cells, held at its potential: the inflow through that
            face, k_table (Ψ_table - Ψ), k_table = K A / (h/2). Its depth is the grid's, whatever the cells' size.
            """
            return water_table_conductance * (water_table_potential - water_potential)

        @boundary_condition("node", "neumann", field="water_potential", where={"boundary_low_z": ">0"})
        def _to_the_air(self, water_potential, air_water_potential, vapour_conductance):
            """Liquid to vapour at the surface: the inflow k_vap (Ψ_air - Ψ), negative (water leaves)."""
            return vapour_conductance * (air_water_potential - water_potential)

        @boundary_condition("node", "neumann", field="water_potential")
        def _taken_by_the_plants(self, plant_uptake):
            return -plant_uptake

        @graph_output("evaporation", location="node", where={"boundary_low_z": ">0"})
        def _evaporation(self, water_potential, air_water_potential, vapour_conductance):
            return vapour_conductance * (water_potential - air_water_potential)


@dataclass
class SoilStructure(StructuralComponent):
    """
    The soil grid's hydraulics: face conductances k = K · A / d, K the harmonic mean of the two voxels' K_sat (which
    vary between voxels, by layer); the surface layer evaporates (a soil-surface vapour conductance), the bottom
    layer's bottom face is the water table (k_table = K A / (h/2)). Regular and adaptive grids alike (faces of unequal
    cells: their own A and d); the layers, the cells' areas and heights are the grid's geometric descriptors
    (boundary_low_z, boundary_high_z, cell_section_z, cell_size_z: set again by an adaptive grid after each
    refinement).
    """
    k: float = state_variable(unit="mm3 s-1 MPa-1", description="Hydraulic conductance of a face.", default=0.,
                              location="edge", kind="intensive")
    vapour_conductance: float = state_variable(unit="mm3 s-1 MPa-1",
                                               description="Liquid-vapour conductance of a surface cell.",
                                               default=0., location="cell", kind="intensive")
    # the grid's geometric descriptors (metafspm's: every grid's, set again after each refinement)
    boundary_low_z: float = input_variable(unit="adim", description="1 at the surface (the grid's top layer).",
                                           by="grid", default=0., location="cell", kind="descriptor")
    boundary_high_z: float = input_variable(unit="adim", description="1 at the water table (the grid's bottom layer).",
                                            by="grid", default=0., location="cell", kind="descriptor")
    cell_section_z: float = input_variable(unit="m2", description="Horizontal area of a cell (its face on the surface "
                                                                  "or the water table).", by="grid", default=0.,
                                           location="cell", kind="descriptor")
    cell_size_z: float = input_variable(unit="m", description="A cell's height.", by="grid", default=0.,
                                        location="cell", kind="descriptor")
    K_sat: float = parameter(unit="mm3 m-1 s-1 MPa-1", description="Hydraulic conductivity of a voxel.",
                             by="SoilStructure", default=5., location="cell")
    soil_surface_conductance: float = parameter(unit="mol m-2 s-1",
                                                description="Vapour conductance of the soil surface.",
                                                by="SoilStructure", default=0.005)
    vapour_factor: float = parameter(unit="mm3 s-1 MPa-1 per (mol m-2 s-1 m2)",
                                     description="e_sat V_w² / (R T P): from a vapour conductance to a liquid one.",
                                     by="SoilStructure", default=VAPOUR_FACTOR)
    water_table_conductance: float = state_variable(unit="mm3 s-1 MPa-1", description="Conductance between a bottom "
                                                                                      "cell's centre and the water "
                                                                                      "table.", default=0.,
                                                    location="cell", kind="intensive")

    @postsegmentation
    def _k(self, K_sat, face_area, face_distance):
        view = self.data_structure.to_graph_view()                       # faces as edges, cells as nodes
        tail, head = view.tail, view.head
        K = np.asarray(K_sat).reshape(-1)
        K_face = 2. * K[tail] * K[head] / np.maximum(K[tail] + K[head], 1e-30)
        return K_face * face_area / face_distance

    @postsegmentation(where={"boundary_low_z": ">0"})
    def _vapour_conductance(self, cell_section_z, soil_surface_conductance, vapour_factor):
        return soil_surface_conductance * cell_section_z * vapour_factor

    @postsegmentation(where={"boundary_high_z": ">0"})
    def _water_table_conductance(self, K_sat, cell_section_z, cell_size_z):
        """K A / (h/2): from the bottom cells' centre to their bottom face."""
        return K_sat * cell_section_z / (cell_size_z / 2.)


SOIL_TRANSLATOR = Translator().link("SoilWaterTransport", "conductance", "SoilStructure", {"k": 1.})


def set_conductivity(grid, topsoil_K, subsoil_K, topsoil_depth):
    """K_sat by layer, from the depth of each cell's centre (any grid)."""
    grid.set("K_sat", np.where(grid.cell_centers()[:, 2] < topsoil_depth, topsoil_K, subsoil_K))


class Soil(EnvironmentModel):
    """
    The soil environment model: a fixed grid over the stand, z pointing down from the surface (5 cm cells by
    default), its conductivity varying by layer (a denser subsoil), the water table at the bottom.
    """
    compartment = "Soil"

    def __init__(self, plant_data_structures, x_range, y_range, time_step, depth=0.4, voxel=0.05,
                 topsoil_K=1.5, subsoil_K=0.6, topsoil_depth=0.2):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        self.grid = self.build_grid(voxel, depth)
        self.structure = SoilStructure(data_structure=self.grid, time_step=time_step)
        self.transport = SoilWaterTransport(data_structure=self.grid, time_step=time_step)
        self.couple(self.structure, self.transport, translator=SOIL_TRANSLATOR)
        self.layers = dict(topsoil_K=topsoil_K, subsoil_K=subsoil_K, topsoil_depth=topsoil_depth)
        set_conductivity(self.grid, **self.layers)
