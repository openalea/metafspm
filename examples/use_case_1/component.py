"""
Components of use case 1 (a wheat plant in its soil and its light): the soil, the transient water flow of
Richards' equation on a grid.

    RichardsSoil        the water in the soil grid: Richards' equation in its mixed form, the pressure head h as the
                        cells' unknown, van Genuchten-Mualem (or Gardner) soils per cell (per layer), the fluxes on
                        the faces with gravity, the top a flux (rain minus evaporation, both limited by the soil), the
                        bottom free drainage, a water table or no flux, the sides periodic (the grid's); the roots'
                        uptake a sink of the cells
    RichardsSoilModel   the environment model of a Scene: the grid over the scene's pattern (adaptive by default),
                        refined where the roots take up water and where h changes steeply, the water carried over as
                        a volume (θ V, not h) when the grid changes
    set_soil_layers, LOAM, SOILS
                        soil hydraulic parameters by layer (Carsel & Parrish 1988's classes, a loam by default)
    scene_translator    the scene links of the soil with a plant water model (MECHA's): the uptake into the cells,
                        the soil's water potential back to the roots

and the plant's own parts, between its shoot model (Adel's) and its root model (ArchiSimple's):

    WheatCrown          an initiator, first in the plant: the seed at the sowing depth and the subcrown internode up
                        to the crown, below the soil surface; the shoot's main stem grows on the crown's top
                        (Adel's shoot_anchor), the seminal roots on the seed (ArchiSimple's root_anchor); it draws
                        its elements (the grain, the subcrown internode), joining the roots and the shoot drawn
    NodalRootEmergence  the rule of the nodal roots (a rule component, "sites"): N nodal roots per leaf appeared, on
                        its axis' phytomer offset ranks below, D °Cd after the leaf's appearance, asked of the root
                        model as a cumulative count per phytomer (nodal_axes_requested), from the phytomer's node
                        (site_x, site_y, site_z: on the crown in the tillering zone)

Coordinates: z points down from the surface (the depth), as on every metafspm soil grid. The total head is h - z,
and gravity pulls the water towards increasing z.

Units: lengths in m, times in s, the pressure head in m of water, the water potential in MPa, volumes in m3 (fluxes
in m3 s-1 on faces and cells; the rainfall and evaporation as m s-1 per area of soil surface), water contents in
m3 m-3, conductivities in m s-1. On the plant, z points up (the planting's: minus the sowing depth for a seed),
thermal times are in °Cd.
"""
from dataclasses import dataclass, fields

import numpy as np
from scipy.sparse import csr_matrix, diags

from openalea.metafspm.coupling.component import (FunctionalComponent, StructuralComponent, input_variable,
                                                  parameter, planting_variable, setting, state_variable)
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.solve.conservation import Conservation
from openalea.metafspm.solve.decorator import (boundary_condition, graph_jacobian, graph_output, graph_system,
                                               node_balance, state)

WATER_DENSITY, GRAVITY = 1000., 9.81                          # kg m-3, m s-2
# The soil's water, conserved by its solve: held θ V, in through the surface, out through the bottom and to the roots
WATER = Conservation(amount=("water_content", "volume"), inflow="surface_inflow", outflow=("drainage", "plant_uptake"),
                     unit="m3")
MPA_PER_METRE = WATER_DENSITY * GRAVITY * 1e-6                # MPa per m of pressure head

VAN_GENUCHTEN_REFERENCES = ("van Genuchten M.Th. (1980) A closed-form equation for predicting the hydraulic "
                            "conductivity of unsaturated soils. Soil Sci. Soc. Am. J. 44: 892-898; Mualem Y. (1976) "
                            "A new model for predicting the hydraulic conductivity of unsaturated porous media. "
                            "Water Resour. Res. 12: 513-522")
VAN_GENUCHTEN_DOI = ["10.2136/sssaj1980.03615995004400050002x", "10.1029/WR012i003p00513"]
CARSEL_PARRISH = ("Carsel R.F., Parrish R.S. (1988) Developing joint probability distributions of soil water "
                  "retention characteristics. Water Resour. Res. 24: 755-769 (the loam's class means)")
CARSEL_PARRISH_DOI = ["10.1029/WR024i005p00755"]
GARDNER = ("Gardner W.R. (1958) Some steady-state solutions of the unsaturated moisture flow equation with "
           "application to evaporation from a water table. Soil Science 85: 228-232")
GARDNER_DOI = ["10.1097/00010694-195804000-00006"]


# ═══════════════════════════════════════════════════════════════════════════════
# Soil hydraulic properties: θ(h), K(h) and their derivatives
# ═══════════════════════════════════════════════════════════════════════════════

def van_genuchten_mualem(pressure_head, residual_water_content, saturated_water_content, alpha, n,
                         saturated_conductivity, pore_connectivity):
    """
    van Genuchten's retention curve and Mualem's conductivity, with their derivatives in h, cell by cell:

        Se = (1 + (α |h|)^n)^-m, m = 1 - 1/n,  θ = θr + (θs - θr) Se,  K = Ks Se^l (1 - (1 - Se^(1/m))^m)^2

    for h < 0; θ = θs and K = Ks for h >= 0. Returns (θ, dθ/dh, K, dK/dh). With n < 2, dK/dh grows without bound
    as h tends to 0 (Mualem's curve), finite at any h < 0.
    """
    h = np.asarray(pressure_head, dtype=float)
    m = 1. - 1. / n
    unsaturated = h < 0.
    x = np.where(unsaturated, -alpha * h, 1.)                         # α |h|, 1 where saturated (masked below)
    x_power_n = x ** n
    one_plus = 1. + x_power_n
    effective_saturation = np.where(unsaturated, one_plus ** -m, 1.)
    saturation_slope = np.where(unsaturated, alpha * m * n * x ** (n - 1.) * one_plus ** (-m - 1.), 0.)
    mualem = np.where(unsaturated, 1. - (x_power_n / one_plus) ** m, 1.)
    mualem_slope = np.where(unsaturated, alpha * m * n * x ** (n - 2.) * one_plus ** (-m - 1.), 0.)
    water_content = residual_water_content + (saturated_water_content - residual_water_content) * effective_saturation
    capacity = (saturated_water_content - residual_water_content) * saturation_slope
    conductivity = saturated_conductivity * effective_saturation ** pore_connectivity * mualem ** 2
    conductivity_slope = saturated_conductivity * (
        pore_connectivity * effective_saturation ** (pore_connectivity - 1.) * saturation_slope * mualem ** 2
        + effective_saturation ** pore_connectivity * 2. * mualem * mualem_slope)
    return water_content, capacity, conductivity, np.where(unsaturated, conductivity_slope, 0.)


def gardner(pressure_head, residual_water_content, saturated_water_content, alpha, saturated_conductivity):
    """
    Gardner's exponential soil, θ = θr + (θs - θr) e^(α h) and K = Ks e^(α h) for h < 0 (θs and Ks above): the
    soil whose Richards equation is linear in K, solved exactly by Srivastava & Yeh (1991). Returns (θ, dθ/dh, K,
    dK/dh).
    """
    h = np.asarray(pressure_head, dtype=float)
    unsaturated = h < 0.
    relative = np.exp(alpha * np.minimum(h, 0.))
    water_content = residual_water_content + (saturated_water_content - residual_water_content) * relative
    capacity = np.where(unsaturated, (saturated_water_content - residual_water_content) * alpha * relative, 0.)
    conductivity = saturated_conductivity * relative
    return water_content, capacity, conductivity, np.where(unsaturated, alpha * conductivity, 0.)


def pressure_head_of(water_content, residual_water_content, saturated_water_content, alpha, n,
                     retention_model="van_genuchten"):
    """The inverse retention curve: h from θ (0 at saturation, θ >= θs); θ is kept above θr."""
    span = saturated_water_content - residual_water_content
    effective_saturation = np.clip((np.asarray(water_content, dtype=float) - residual_water_content) / span,
                                   np.finfo(float).tiny, 1.)
    if retention_model == "gardner":
        return np.log(effective_saturation) / alpha
    m = 1. - 1. / n
    return -(effective_saturation ** (-1. / m) - 1.) ** (1. / n) / alpha


# ═══════════════════════════════════════════════════════════════════════════════
# The soil classes and layers
# ═══════════════════════════════════════════════════════════════════════════════

# Carsel & Parrish (1988)'s class means (α in m-1, Ks in m s-1); the pore connectivity l = 0.5 (Mualem)
SOILS = {
    "sand":       dict(residual_water_content=0.045, saturated_water_content=0.43, van_genuchten_alpha=14.5,
                       van_genuchten_n=2.68, saturated_conductivity=712.8e-2 / 86400.),
    "loamy_sand": dict(residual_water_content=0.057, saturated_water_content=0.41, van_genuchten_alpha=12.4,
                       van_genuchten_n=2.28, saturated_conductivity=350.2e-2 / 86400.),
    "sandy_loam": dict(residual_water_content=0.065, saturated_water_content=0.41, van_genuchten_alpha=7.5,
                       van_genuchten_n=1.89, saturated_conductivity=106.1e-2 / 86400.),
    "loam":       dict(residual_water_content=0.078, saturated_water_content=0.43, van_genuchten_alpha=3.6,
                       van_genuchten_n=1.56, saturated_conductivity=24.96e-2 / 86400.),
    "silt_loam":  dict(residual_water_content=0.067, saturated_water_content=0.45, van_genuchten_alpha=2.0,
                       van_genuchten_n=1.41, saturated_conductivity=10.8e-2 / 86400.),
    "clay_loam":  dict(residual_water_content=0.095, saturated_water_content=0.41, van_genuchten_alpha=1.9,
                       van_genuchten_n=1.31, saturated_conductivity=6.24e-2 / 86400.),
}
LOAM = SOILS["loam"]


def set_soil_layers(grid, layers):
    """
    The soil's parameters by layer, from the depth of each cell's centre (any grid): *layers* is a sequence of
    (bottom depth in m, {parameter: value}), from the surface down, the last layer reaching the bottom whatever its
    depth. A parameter a layer does not give keeps its value.
    """
    depth = grid.cell_centers()[:, 2]
    top = -np.inf
    for index, (bottom, values) in enumerate(layers):
        chosen = (depth >= top) & ((depth < bottom) | (index == len(layers) - 1))
        for name, value in values.items():
            current = np.array(grid.get(name), dtype=float).reshape(-1)
            current[chosen] = value
            grid.set(name, current.reshape(np.shape(grid.get(name))))
        top = bottom


# ═══════════════════════════════════════════════════════════════════════════════
# The face fluxes and the boundary fluxes, with their derivatives (shared by the equations and their Jacobian)
# ═══════════════════════════════════════════════════════════════════════════════

def _properties(component, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                van_genuchten_n, saturated_conductivity, pore_connectivity):
    """(θ, dθ/dh, K, dK/dh) of each cell, by the component's retention model."""
    if component.retention_model == "gardner":
        return gardner(pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                       saturated_conductivity)
    return van_genuchten_mualem(pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                                van_genuchten_n, saturated_conductivity, pore_connectivity)


def _face_fluxes(component, pressure_head, conductivity, conductivity_slope, face_area, face_distance, gravity):
    """
    The flux through each face, from its tail cell to its head cell (towards increasing coordinates, down along z),

        q = K_f A (Δh / d + g),  Δh = h_tail - h_head,

    g = 1 on the z faces (gravity), 0 on the others, and its derivatives in h: (q, dq/dh as a faces x cells matrix).
    Δh is the grid's consistent edge difference (exact for a linear h between cells of different sizes). K_f is the
    upstream cell's K (the cell the water comes from), or the arithmetic mean of the two cells' K (the
    face_conductivity setting).
    """
    view = component.graph_view
    difference = view.edge_difference
    tail, head = view.tail, view.head
    drive = np.asarray(difference @ pressure_head).reshape(-1) / face_distance + gravity
    faces = np.arange(tail.size)
    if component.face_conductivity == "upstream":
        upstream = np.where(drive >= 0., tail, head)
        face_conductivity = conductivity[upstream]
        conductivity_derivative = csr_matrix((conductivity_slope[upstream], (faces, upstream)),
                                             shape=(tail.size, pressure_head.size))
    else:
        face_conductivity = 0.5 * (conductivity[tail] + conductivity[head])
        conductivity_derivative = csr_matrix((np.r_[0.5 * conductivity_slope[tail], 0.5 * conductivity_slope[head]],
                                              (np.r_[faces, faces], np.r_[tail, head])),
                                             shape=(tail.size, pressure_head.size))
    flux = face_conductivity * face_area * drive
    derivative = (diags(face_conductivity * face_area / face_distance) @ difference
                  + diags(face_area * drive) @ conductivity_derivative)
    return flux, derivative.tocsr()


def _boundary_conductivity(component, conductivity, conductivity_slope, boundary_conductivity, cell_upstream):
    """K between a cell and a boundary held at a head whose K is *boundary_conductivity*: the upstream one, or the
    mean; with its derivative in the cell's h."""
    if component.face_conductivity == "upstream":
        return (np.where(cell_upstream, conductivity, boundary_conductivity),
                np.where(cell_upstream, conductivity_slope, 0.))
    return 0.5 * (conductivity + boundary_conductivity), 0.5 * conductivity_slope


def _surface_inflow(component, pressure_head, conductivity, conductivity_slope, saturated_conductivity,
                    boundary_area, half_height, rainfall, potential_evaporation, surface_minimum_head,
                    surface_conductivity):
    """
    The water entering a surface cell through the soil surface (m3 s-1, negative when it evaporates), and its
    derivative in h. The potential flux (rainfall - potential_evaporation) A is limited by what the soil takes or
    gives, the Darcy flux between the surface and the cell's centre (half a cell below):

      * infiltrating, the surface at most ponded at h = 0 (the excess runs off), K(0) = Ks:
        capacity = K_f A ((0 - h) / (Δz/2) + 1);
      * evaporating, the surface at most as dry as surface_minimum_head:
        capacity = K_f A ((h - h_min) / (Δz/2) - 1).

    Returns (inflow, d inflow / dh, runoff).
    """
    potential = (rainfall - potential_evaporation) * boundary_area
    # infiltration: the water comes from the surface (upstream: the surface's K, at h = 0)
    infiltration_conductivity, infiltration_slope = _boundary_conductivity(
        component, conductivity, conductivity_slope, saturated_conductivity, np.zeros(np.shape(conductivity), bool))
    infiltration_drive = -pressure_head / half_height + 1.
    infiltration_capacity = infiltration_conductivity * boundary_area * infiltration_drive
    infiltration_capacity_slope = (-infiltration_conductivity * boundary_area / half_height
                                   + infiltration_slope * boundary_area * infiltration_drive)
    # evaporation: the water comes from the cell (upstream: the cell's K)
    evaporation_conductivity, evaporation_slope = _boundary_conductivity(
        component, conductivity, conductivity_slope, surface_conductivity, np.ones(np.shape(conductivity), bool))
    evaporation_drive = (pressure_head - surface_minimum_head) / half_height - 1.
    evaporation_capacity = evaporation_conductivity * boundary_area * evaporation_drive
    evaporation_capacity_slope = (evaporation_conductivity * boundary_area / half_height
                                  + evaporation_slope * boundary_area * evaporation_drive)
    infiltrating = potential >= 0.
    limited_infiltration = infiltrating & (infiltration_capacity < potential)
    limited_evaporation = ~infiltrating & (np.maximum(evaporation_capacity, 0.) < -potential)
    inflow = np.where(limited_infiltration, infiltration_capacity,
                      np.where(limited_evaporation, -np.maximum(evaporation_capacity, 0.), potential))
    slope = np.where(limited_infiltration, infiltration_capacity_slope,
                     np.where(limited_evaporation & (evaporation_capacity > 0.), -evaporation_capacity_slope, 0.))
    runoff = np.where(infiltrating, potential - inflow, 0.)
    return inflow, slope, runoff


def _bottom_outflow(component, pressure_head, conductivity, conductivity_slope, boundary_area, half_height,
                    water_table_head, water_table_conductivity):
    """
    The water leaving a bottom cell through the grid's bottom face (m3 s-1), and its derivative in h, by the
    bottom_boundary setting:

      * "free_drainage": a unit gradient of the total head, q = K(h) A;
      * "water_table": the bottom face held at water_table_head (0: the water table at the bottom),
        q = K_f A ((h - h_table) / (Δz/2) + 1);
      * "no_flux": 0.
    """
    if component.bottom_boundary == "free_drainage":
        return conductivity * boundary_area, conductivity_slope * boundary_area
    if component.bottom_boundary == "no_flux":
        return np.zeros(np.shape(pressure_head)), np.zeros(np.shape(pressure_head))
    drive = (pressure_head - water_table_head) / half_height + 1.
    face_conductivity, face_slope = _boundary_conductivity(component, conductivity, conductivity_slope,
                                                           water_table_conductivity, drive >= 0.)
    return (face_conductivity * boundary_area * drive,
            face_conductivity * boundary_area / half_height + face_slope * boundary_area * drive)


# ═══════════════════════════════════════════════════════════════════════════════
# The component
# ═══════════════════════════════════════════════════════════════════════════════

_SOIL = ("residual_water_content", "saturated_water_content", "van_genuchten_alpha", "van_genuchten_n",
         "saturated_conductivity", "pore_connectivity")


@dataclass
class RichardsSoil(FunctionalComponent):
    """
    The transient water flow in a soil grid, Richards' equation in its mixed form (Celia et al. 1990), on an
    ArrayDataStructure or an AdaptiveGridDataStructure: the cells are the nodes, the faces the edges, the pressure
    head h each cell's unknown. Over a (sub-)step of length dt, each cell's balance is

        (θ(h) - θ(h_prev)) V + dt (Σ_faces q + uptake - inflow_boundaries) = 0,

    written per cell volume (a change of water content, m3 m-3, so that Newton's tolerance is one on the water
    contents and the balance closes to it), the face fluxes q = K_f A (Δh/d + g) (gravity along z, pointing down),
    K_f upstream-weighted or the arithmetic mean (face_conductivity). The soil is van Genuchten-Mualem's (or
    Gardner's, retention_model) with its parameters per cell, set by layers (set_soil_layers). The top: the rainfall
    minus the potential evaporation (inputs; when forcings of their names are given, by a Scene or set_forcings, read
    at the end of each sub-step, so that a rain varying within the step is followed), limited by what the soil takes
    or gives (the excess rain runs off); the bottom: free drainage, a water table or no flux (bottom_boundary); the
    sides: the grid's (periodic over the scene's pattern). The roots' uptake (plant_uptake, summed into the cells
    by the scene) is a sink.

    Solved by Newton with its analytic Jacobian (dθ/dh, dK/dh) and an Armijo line search (multi-day steps through
    a dry soil), to 1e-13 of the water contents, over the time step by step doubling (integrate="adaptive",
    rtol 1e-3 and atol 1e-4 m on h). The water is declared conserved (graph_system(conserved=...)): over each
    call, the water held Σ θ V, what entered through the surface and what left through the bottom and to the roots,
    integrated over the accepted sub-steps by metafspm, so that the balance is checked at every step
    (Component.conservation(), the recorder's conservation.csv): its residual is the solver's tolerance.

    When an adaptive grid changes, the water is carried over as a volume (θ, volume-averaged, so θ V is conserved)
    and h is recomputed from it (carry_over()): RichardsSoilModel does it after each refinement.
    """
    # the unknowns and the water state
    pressure_head: float = state_variable(unit="m", description="Pressure head of the soil water (negative: "
                                                                "suction).", default=-1., location="cell",
                                          kind="intensive")
    water_content: float = state_variable(unit="m3 m-3", description="Volumetric water content θ(h).", default=0.3,
                                          location="cell", kind="intensive")
    water_potential: float = state_variable(unit="MPa", description="Matric potential of the soil water, ρ g h.",
                                            default=-0.00981, location="cell", kind="intensive")
    hydraulic_conductivity: float = state_variable(unit="m s-1", description="Hydraulic conductivity K(h).",
                                                   default=0., location="cell", kind="intensive")
    water_flux: float = state_variable(unit="m3 s-1", description="Flux through a face, towards increasing "
                                                                  "coordinates (down along z), at the end of the "
                                                                  "last sub-step.", default=0., location="edge",
                                       kind="extensive")
    surface_inflow: float = state_variable(unit="m3 s-1", description="Water entering a surface cell through the "
                                                                      "soil surface (negative: evaporation).",
                                           default=0., location="cell", kind="extensive")
    runoff: float = state_variable(unit="m3 s-1", description="Rain the surface cell could not take.", default=0.,
                                   location="cell", kind="extensive")
    drainage: float = state_variable(unit="m3 s-1", description="Water leaving a bottom cell through the grid's "
                                                                "bottom face.", default=0., location="cell",
                                     kind="extensive")
    # the grid's geometry (set_geometry: at construction and after each refinement)
    volume: float = state_variable(unit="m3", description="Volume of a cell.", default=0., location="cell",
                                   kind="extensive")
    is_surface: float = state_variable(unit="adim", description="1 in the cells at the soil surface.", default=0.,
                                       location="cell", kind="descriptor")
    is_bottom: float = state_variable(unit="adim", description="1 in the cells at the grid's bottom.", default=0.,
                                      location="cell", kind="descriptor")
    boundary_area: float = state_variable(unit="m2", description="Horizontal area of a cell (its face on the soil "
                                                                 "surface or the bottom).", default=0.,
                                          location="cell", kind="extensive")
    half_height: float = state_variable(unit="m", description="Half a cell's height: from its centre to its top or "
                                                              "bottom face.", default=0., location="cell",
                                        kind="descriptor")
    gravity: float = state_variable(unit="adim", description="1 on the z faces (gravity along the face), 0 on the "
                                                             "others.", default=0., location="edge",
                                    kind="descriptor")
    # inputs
    plant_uptake: float = input_variable(unit="m3 s-1", description="Water taken up by the roots from a cell.",
                                         by="plants", default=0., location="cell", kind="extensive")
    rainfall: float = input_variable(unit="m s-1", description="Rain (and irrigation) reaching the soil surface "
                                                               "(a forcing of this name, read at each sub-step, "
                                                               "else the input).", by="forcings", default=0.)
    potential_evaporation: float = input_variable(unit="m s-1", description="Potential evaporation of the soil "
                                                                            "surface (a forcing of this name, read "
                                                                            "at each sub-step, else the input).",
                                                  by="forcings", default=0.)
    # the soil, per cell (set_soil_layers); a loam by default
    residual_water_content: float = parameter(unit="m3 m-3", description="Residual water content θr.",
                                              by="RichardsSoil", default=LOAM["residual_water_content"],
                                              location="cell", min_value=0., max_value=0.25,
                                              references=CARSEL_PARRISH, DOI=CARSEL_PARRISH_DOI)
    saturated_water_content: float = parameter(unit="m3 m-3", description="Saturated water content θs.",
                                               by="RichardsSoil", default=LOAM["saturated_water_content"],
                                               location="cell", min_value=0.25, max_value=0.6,
                                               references=CARSEL_PARRISH, DOI=CARSEL_PARRISH_DOI)
    van_genuchten_alpha: float = parameter(unit="m-1", description="van Genuchten's α, the inverse of the "
                                                                   "air-entry head (Gardner's α too).",
                                           by="RichardsSoil", default=LOAM["van_genuchten_alpha"], location="cell",
                                           min_value=0.5, max_value=20., references=VAN_GENUCHTEN_REFERENCES,
                                           DOI=VAN_GENUCHTEN_DOI)
    van_genuchten_n: float = parameter(unit="adim", description="van Genuchten's n (m = 1 - 1/n).",
                                       by="RichardsSoil", default=LOAM["van_genuchten_n"], location="cell",
                                       min_value=1.05, max_value=3., references=VAN_GENUCHTEN_REFERENCES,
                                       DOI=VAN_GENUCHTEN_DOI)
    saturated_conductivity: float = parameter(unit="m s-1", description="Saturated hydraulic conductivity Ks.",
                                              by="RichardsSoil", default=LOAM["saturated_conductivity"],
                                              location="cell", min_value=1e-8, max_value=1e-4,
                                              references=CARSEL_PARRISH, DOI=CARSEL_PARRISH_DOI)
    pore_connectivity: float = parameter(unit="adim", description="Mualem's pore connectivity (tortuosity) l.",
                                         by="RichardsSoil", default=0.5, location="cell", min_value=-2.,
                                         max_value=2., references=VAN_GENUCHTEN_REFERENCES, DOI=VAN_GENUCHTEN_DOI)
    # the boundaries
    water_table_head: float = parameter(unit="m", description="Pressure head at the grid's bottom face with "
                                                              "bottom_boundary 'water_table' (0: the water table "
                                                              "there; positive: above it).", by="RichardsSoil",
                                        default=0., min_value=-5., max_value=5.)
    surface_minimum_head: float = parameter(unit="m", description="The driest pressure head of the soil surface: "
                                                                  "the evaporation is limited to what the soil "
                                                                  "brings to it (Hydrus' hCritA).",
                                            by="RichardsSoil", default=-100., min_value=-1000., max_value=-10.,
                                            references="Šimůnek J., van Genuchten M.Th., Šejna M. (2016) Recent "
                                                       "developments and applications of the HYDRUS computer "
                                                       "software packages. Vadose Zone J. 15(7)",
                                            DOI=["10.2136/vzj2016.04.0033"])
    # settings
    face_conductivity: str = setting(description="The conductivity of a face: 'upstream' (the cell the water comes "
                                                 "from, robust for infiltration fronts) or 'arithmetic' (the mean).",
                                     default="upstream", choices=("upstream", "arithmetic"),
                                     references="Forsyth P.A., Wu Y.S., Pruess K. (1995) Robust numerical methods "
                                                "for saturated-unsaturated flow with dry initial conditions in "
                                                "heterogeneous media. Adv. Water Resour. 18: 25-38",
                                     DOI=["10.1016/0309-1708(95)00020-J"])
    retention_model: str = setting(description="The soil's θ(h) and K(h): 'van_genuchten' (van Genuchten-Mualem) "
                                               "or 'gardner' (exponential).", default="van_genuchten",
                                   choices=("van_genuchten", "gardner"), references=GARDNER, DOI=GARDNER_DOI)
    bottom_boundary: str = setting(description="The grid's bottom: 'free_drainage' (a unit gradient), "
                                               "'water_table' (held at water_table_head) or 'no_flux'.",
                                   default="free_drainage", choices=("free_drainage", "water_table", "no_flux"))

    def __post_init__(self):
        super().__post_init__()
        self.set_geometry()
        self.update_water_content()

    # ── the grid ──────────────────────────────────────────────────────────────

    def set_geometry(self):
        """
        The surface and bottom cells, their horizontal areas and half heights, the cells' volumes and the z faces,
        from the grid's geometry: a regular grid or an adaptive one, whose refinement calls it again. The grid has
        three axes, z the third (a column is a (1, 1, n) grid).
        """
        grid = self.data_structure
        size = grid.cell_sizes()
        cells = size.shape[0]
        grid.set("is_surface", np.asarray(grid.layer_mask(z=0), dtype=float).reshape(np.shape(grid.get("is_surface"))))
        grid.set("is_bottom", np.asarray(grid.layer_mask(z=-1), dtype=float).reshape(np.shape(grid.get("is_bottom"))))
        shape = np.shape(grid.get("volume"))
        grid.set("volume", np.broadcast_to(grid.cell_volume(), (cells,)).reshape(shape))
        grid.set("boundary_area", (size[:, 0] * size[:, 1]).reshape(shape))
        grid.set("half_height", (size[:, 2] / 2.).reshape(shape))
        grid.set("gravity", (np.asarray(grid.face_axis()) == grid.axes.index("z")).astype(float))

    def _cell_values(self, *names):
        return [np.asarray(self.data_structure.get(name), dtype=float).reshape(-1) for name in names]

    def update_water_content(self):
        """θ, K and the water potential from h (after h or the soil's parameters were set outside a solve)."""
        grid = self.data_structure
        (pressure_head,) = self._cell_values("pressure_head")
        water_content, _, conductivity, _ = _properties(self, pressure_head, *self._cell_values(*_SOIL))
        shape = np.shape(grid.get("water_content"))
        grid.set("water_content", water_content.reshape(shape))
        grid.set("hydraulic_conductivity", conductivity.reshape(shape))
        grid.set("water_potential", (MPA_PER_METRE * pressure_head).reshape(shape))

    def set_pressure_head(self, values):
        """h (a number, an array per cell, or a function of the cells' depth), with θ, K and Ψ following."""
        grid = self.data_structure
        depth = grid.cell_centers()[:, 2]
        values = values(depth) if callable(values) else np.broadcast_to(np.asarray(values, dtype=float), depth.shape)
        grid.set("pressure_head", np.asarray(values, dtype=float).reshape(np.shape(grid.get("pressure_head"))))
        self.update_water_content()

    def carry_over(self):
        """
        After the grid changed (refined or coarsened): the grid carried θ over by volume (θ V conserved), h and the
        soil parameters by volume averages. h is recomputed from the carried θ by the inverse retention curve, so
        that the next balance starts from the water the cells hold; a saturated cell (θ = θs) keeps its carried
        h >= 0. The geometry is set again.
        """
        self.set_geometry()
        grid = self.data_structure
        water_content, pressure_head, residual, saturated, alpha, n = self._cell_values(
            "water_content", "pressure_head", "residual_water_content", "saturated_water_content",
            "van_genuchten_alpha", "van_genuchten_n")
        unsaturated = water_content < saturated
        inverse = pressure_head_of(water_content, residual, saturated, alpha, n, self.retention_model)
        head = np.where(unsaturated, inverse, np.maximum(pressure_head, 0.))
        grid.set("pressure_head", head.reshape(np.shape(grid.get("pressure_head"))))

    def stored_water(self):
        """The water the soil holds, Σ θ V (m3)."""
        water_content, volume = self._cell_values("water_content", "volume")
        return float((water_content * volume).sum())

    # ── the equations ─────────────────────────────────────────────────────────

    @graph_system(node_unknowns=["pressure_head"], solver="newton", transient=True,
                  integrate="adaptive", rtol=1e-3, atol=1e-4, tolerance=1e-13, max_iterations=30, linesearch=True,
                  schedule_as="state", conserved={"water": WATER})
    class _water:
        """Richards' equation in mixed form, per cell volume."""

        @node_balance(field="pressure_head")
        def _balance(self, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                     van_genuchten_n, saturated_conductivity, pore_connectivity, face_area, face_distance, gravity,
                     plant_uptake, volume):
            """(θ(h) - θ(h_prev)) + dt / V (Σ q + uptake); the boundaries' inflows are subtracted by the conditions."""
            soil = (residual_water_content, saturated_water_content, van_genuchten_alpha, van_genuchten_n,
                    saturated_conductivity, pore_connectivity)
            water_content, _, conductivity, conductivity_slope = _properties(self, pressure_head, *soil)
            previous_content = _properties(self, self.previous("pressure_head"), *soil)[0]
            flux, _ = _face_fluxes(self, pressure_head, conductivity, conductivity_slope, face_area, face_distance,
                                   gravity)
            outflow = np.asarray(self.graph_view.incidence @ flux).reshape(-1)
            return water_content - previous_content + self.dt / volume * (outflow + plant_uptake)

        @boundary_condition("node", "neumann", field="pressure_head", filters={"is_surface": ">0"})
        def _soil_surface(self, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                          van_genuchten_n, saturated_conductivity, pore_connectivity, boundary_area, half_height,
                          rainfall, potential_evaporation, surface_minimum_head, volume):
            """The rain minus the evaporation entering through the surface, over the sub-step per cell volume."""
            return self.dt / volume * _surface(self, pressure_head, residual_water_content, saturated_water_content,
                                               van_genuchten_alpha, van_genuchten_n, saturated_conductivity,
                                               pore_connectivity, boundary_area, half_height, rainfall,
                                               potential_evaporation, surface_minimum_head)[0]

        @boundary_condition("node", "neumann", field="pressure_head", filters={"is_bottom": ">0"})
        def _grid_bottom(self, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                         van_genuchten_n, saturated_conductivity, pore_connectivity, boundary_area, half_height,
                         water_table_head, volume):
            """Free drainage, the water table or no flux, over the sub-step per cell volume (an inflow: minus the
            outflow)."""
            return -self.dt / volume * _bottom(self, pressure_head, residual_water_content, saturated_water_content,
                                               van_genuchten_alpha, van_genuchten_n, saturated_conductivity,
                                               pore_connectivity, boundary_area, half_height, water_table_head)[0]

        @graph_jacobian
        def _jacobian(self, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                      van_genuchten_n, saturated_conductivity, pore_connectivity, face_area, face_distance, gravity,
                      boundary_area, half_height, rainfall, potential_evaporation, surface_minimum_head,
                      water_table_head, is_surface, is_bottom, volume):
            """
            The balance's derivative in h: dθ/dh on the diagonal, dt / V B dq/dh (dq/dh through Δh and K_f(h)),
            minus the boundaries' inflow slopes.
            """
            soil = (residual_water_content, saturated_water_content, van_genuchten_alpha, van_genuchten_n,
                    saturated_conductivity, pore_connectivity)
            _, capacity, conductivity, conductivity_slope = _properties(self, pressure_head, *soil)
            _, flux_derivative = _face_fluxes(self, pressure_head, conductivity, conductivity_slope, face_area,
                                              face_distance, gravity)
            arguments = (pressure_head, *soil, boundary_area, half_height)
            inflow_slope = _surface(self, *arguments, rainfall, potential_evaporation, surface_minimum_head)[1]
            outflow_slope = _bottom(self, *arguments, water_table_head)[1]
            exchange_slope = (is_surface > 0) * inflow_slope - (is_bottom > 0) * outflow_slope
            scale = diags(self.dt / volume)
            return (diags(capacity) + scale @ self.graph_view.incidence @ flux_derivative
                    - diags(self.dt / volume * exchange_slope)).tocsr()

        @graph_output("water_content", location="node")
        def _water_content(self, pressure_head, residual_water_content, saturated_water_content,
                           van_genuchten_alpha, van_genuchten_n, saturated_conductivity, pore_connectivity):
            return _properties(self, pressure_head, residual_water_content, saturated_water_content,
                               van_genuchten_alpha, van_genuchten_n, saturated_conductivity, pore_connectivity)[0]

        @graph_output("hydraulic_conductivity", location="node")
        def _hydraulic_conductivity(self, pressure_head, residual_water_content, saturated_water_content,
                                    van_genuchten_alpha, van_genuchten_n, saturated_conductivity, pore_connectivity):
            return _properties(self, pressure_head, residual_water_content, saturated_water_content,
                               van_genuchten_alpha, van_genuchten_n, saturated_conductivity, pore_connectivity)[2]

        @graph_output("water_potential", location="node")
        def _water_potential(self, pressure_head):
            return MPA_PER_METRE * pressure_head

        @graph_output("water_flux", location="edge")
        def _water_flux(self, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                        van_genuchten_n, saturated_conductivity, pore_connectivity, face_area, face_distance,
                        gravity):
            _, _, conductivity, conductivity_slope = _properties(
                self, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                van_genuchten_n, saturated_conductivity, pore_connectivity)
            return _face_fluxes(self, pressure_head, conductivity, conductivity_slope, face_area, face_distance,
                                gravity)[0]

        @graph_output("surface_inflow", location="node", filters={"is_surface": ">0"})
        def _surface_inflow(self, pressure_head, residual_water_content, saturated_water_content,
                            van_genuchten_alpha, van_genuchten_n, saturated_conductivity, pore_connectivity,
                            boundary_area, half_height, rainfall, potential_evaporation, surface_minimum_head):
            return _surface(self, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                            van_genuchten_n, saturated_conductivity, pore_connectivity, boundary_area, half_height,
                            rainfall, potential_evaporation, surface_minimum_head)[0]

        @graph_output("runoff", location="node", filters={"is_surface": ">0"})
        def _runoff(self, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                    van_genuchten_n, saturated_conductivity, pore_connectivity, boundary_area, half_height, rainfall,
                    potential_evaporation, surface_minimum_head):
            return _surface(self, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                            van_genuchten_n, saturated_conductivity, pore_connectivity, boundary_area, half_height,
                            rainfall, potential_evaporation, surface_minimum_head)[2]

        @graph_output("drainage", location="node", filters={"is_bottom": ">0"})
        def _drainage(self, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                      van_genuchten_n, saturated_conductivity, pore_connectivity, boundary_area, half_height,
                      water_table_head):
            return _bottom(self, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
                           van_genuchten_n, saturated_conductivity, pore_connectivity, boundary_area, half_height,
                           water_table_head)[0]


def _surface(component, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
             van_genuchten_n, saturated_conductivity, pore_connectivity, boundary_area, half_height, rainfall,
             potential_evaporation, surface_minimum_head):
    """The surface inflow of the cells given (inflow, d inflow / dh, runoff), from their soil; the rainfall and the
    potential evaporation the forcings' at the (sub-)step when given, else the inputs."""
    if component.has_forcing("rainfall"):
        rainfall = component.forcing("rainfall")
    if component.has_forcing("potential_evaporation"):
        potential_evaporation = component.forcing("potential_evaporation")
    soil = (residual_water_content, saturated_water_content, van_genuchten_alpha, van_genuchten_n,
            saturated_conductivity, pore_connectivity)
    _, _, conductivity, conductivity_slope = _properties(component, pressure_head, *soil)
    surface_conductivity = _properties(component, np.broadcast_to(surface_minimum_head, np.shape(pressure_head)),
                                       *soil)[2]
    return _surface_inflow(component, pressure_head, conductivity, conductivity_slope, saturated_conductivity,
                           boundary_area, half_height, rainfall, potential_evaporation, surface_minimum_head,
                           surface_conductivity)


def _bottom(component, pressure_head, residual_water_content, saturated_water_content, van_genuchten_alpha,
            van_genuchten_n, saturated_conductivity, pore_connectivity, boundary_area, half_height,
            water_table_head):
    """The bottom outflow of the cells given (outflow, d outflow / dh), from their soil."""
    soil = (residual_water_content, saturated_water_content, van_genuchten_alpha, van_genuchten_n,
            saturated_conductivity, pore_connectivity)
    _, _, conductivity, conductivity_slope = _properties(component, pressure_head, *soil)
    water_table_conductivity = _properties(component, np.broadcast_to(water_table_head, np.shape(pressure_head)),
                                           *soil)[2]
    return _bottom_outflow(component, pressure_head, conductivity, conductivity_slope, boundary_area, half_height,
                           water_table_head, water_table_conductivity)


# ═══════════════════════════════════════════════════════════════════════════════
# The environment model, and its links with the plants
# ═══════════════════════════════════════════════════════════════════════════════

class RichardsSoilModel(EnvironmentModel):
    """
    The soil of a Scene: RichardsSoil on a grid over the scene's pattern down to *depth*, periodic in x and y (the
    stand repeated), z down; an AdaptiveGridDataStructure of *voxel* cells refinable max_level times by default
    (adaptive=False: a regular ArrayDataStructure). The soil by layers (*layers*, set_soil_layers: a loam by default),
    h starting at *initial_pressure_head* (m: a number or a function of the depth).

    After each run, an adaptive grid is refined where the roots take up water (the uptake per volume above
    refine_uptake_fraction of its largest value, when that is above uptake_floor, m3 m-3 s-1: not a plant's
    round-off before it transpires, about 1e-21, a seedling's 1e-9) and where h changes steeply (|∇h| above refine_gradient, m m-1:
    a front; 1 at hydrostatic equilibrium), and coarsened where both are low (coarsen_uptake_fraction,
    coarsen_gradient). The water is carried over as a volume (RichardsSoil.carry_over: θ V conserved, h from θ),
    which no metafspm hook does for derived variables: it is done here, after the refinement.

    Its links with the plants (scene_translator): the plants' uptake summed into the cells (plant_uptake, m3 s-1),
    the soil's water potential (water_potential, MPa) broadcast back to their roots.
    """
    compartment = "Soil"

    def __init__(self, plant_data_structures, xrange, yrange, time_step, depth=1., voxel=0.05, adaptive=True,
                 max_level=2, layers=((np.inf, LOAM),), initial_pressure_head=-1., refine_uptake_fraction=0.05,
                 coarsen_uptake_fraction=0.005, refine_gradient=5., coarsen_gradient=2., uptake_floor=1e-15):
        super().__init__(plant_data_structures, xrange, yrange, time_step)
        self.grid = self.build_grid(voxel, depth, adaptive=adaptive, max_level=max_level)
        self.soil = RichardsSoil(data_structure=self.grid, time_step=time_step)
        self.components = [self.soil]
        self.layers = layers
        set_soil_layers(self.grid, layers)
        self.soil.set_pressure_head(initial_pressure_head)
        self.adaptive = adaptive
        self.refine_uptake_fraction, self.coarsen_uptake_fraction = refine_uptake_fraction, coarsen_uptake_fraction
        self.refine_gradient, self.coarsen_gradient = refine_gradient, coarsen_gradient
        self.uptake_floor = uptake_floor
        self.history = []                                    # cells per level after each adaptation
        self._water_set = False

    def full_run(self):
        if not self._water_set:                              # the scenario's parameters are set: θ from h, once
            self.soil.update_water_content()
            self._water_set = True
        super().full_run()

    def after_run(self):
        if self.adaptive:
            self.adapt()

    def uptake_density(self):
        """The roots' uptake per cell volume (m3 s-1 m-3)."""
        uptake, volume = self.soil._cell_values("plant_uptake", "volume")
        return uptake / volume

    def head_gradient(self):
        """Each cell's largest |Δh / d| over its faces (m m-1)."""
        grid = self.grid
        view = grid.to_graph_view()
        (pressure_head,) = self.soil._cell_values("pressure_head")
        gradient = np.abs(np.asarray(view.edge_difference @ pressure_head).reshape(-1)
                          / np.asarray(grid.get("face_distance")).reshape(-1))
        largest = np.zeros(pressure_head.size)
        np.maximum.at(largest, view.tail, gradient)
        np.maximum.at(largest, view.head, gradient)
        return largest

    def adapt(self):
        """Refine and coarsen on the uptake density and the head gradient, then carry the water over."""
        density, gradient = self.uptake_density(), self.head_gradient()
        top = density.max()
        refine = gradient > self.refine_gradient
        coarsen = gradient < self.coarsen_gradient
        if top > self.uptake_floor:                          # an uptake, not round-off (a plant not transpiring)
            refine |= density > self.refine_uptake_fraction * top
            coarsen &= density < self.coarsen_uptake_fraction * top
        coarsenable = set(self.grid.cell_ids()[coarsen & ~refine].tolist())
        changed = self.grid.refine(refine) > 0
        changed |= self.grid.coarsen(np.isin(self.grid.cell_ids(), list(coarsenable))) > 0
        if changed:
            self.soil.carry_over()
        self.history.append(np.bincount(self.grid.refinement_level(), minlength=self.grid.max_level + 1))

    def checkpoint_state(self) -> dict:
        return {**super().checkpoint_state(), "water_set": self._water_set}

    def restore_state(self, state: dict) -> None:
        super().restore_state(state)
        self._water_set = state.get("water_set", True)


ROOT_SURFACE = {"mask": "root_surface", "method": "overlap"}


def scene_translator(plant_component="MechaHydraulics", uptake="root_uptake", uptake_factor=1e-6 / 86400.,
                     soil_potential="soil_psi", potential_factor=1e4, mapping=ROOT_SURFACE) -> Translator:
    """
    The scene links of the soil with a plant water model: its *uptake* (MECHA's root_uptake, cm3 d-1, factor
    1e-6 / 86400 to m3 s-1) summed into the cells' plant_uptake, and the soil's water_potential (MPa, factor 1e4 to
    MECHA's soil_psi in hPa) back to its *soil_potential*. Their *mapping* (Link.mapping, the Scene's CrossMapping
    of the plants onto the grid): the plant model's mask root_surface (the roots' soil nodes), by overlap.
    """
    return (Translator()
            .link(plant_component, soil_potential, "RichardsSoil", {"water_potential": potential_factor},
                  mapping=mapping)
            .link("RichardsSoil", "plant_uptake", plant_component, {uptake: uptake_factor}, mapping=mapping))


# ═══════════════════════════════════════════════════════════════════════════════
# The plant's seed and crown: WheatCrown, an initiator
# ═══════════════════════════════════════════════════════════════════════════════

KIRBY_1993 = ("Kirby E.J.M. (1993) Effect of sowing depth on seedling emergence, growth and development in barley "
              "and wheat. Field Crops Res. 35: 101-111 (the crown forms at about 2 cm below the surface whatever the "
              "sowing depth, the subcrown internode spanning the rest)")
GEGAS_2010 = ("Gegas V.C., Nazari A., Griffiths S., Simmonds J., Fish L., Orford S., Sayers L., Doonan J.H., Snape "
              "J.W. (2010) A genetic framework for grain size and shape variation in wheat. Plant Cell 22: 1046-1056 "
              "(grain lengths about 5.5 to 7.5 mm, widths 2.8 to 3.8 mm)")
GEGAS_2010_DOI = ["10.1105/tpc.110.074153"]


@dataclass
class WheatCrown(StructuralComponent):
    """
    A wheat plant's seed and subcrown internode, the initiator that comes first in the plant (its initiate_plant):
    the structure the shoot and the root models build on, so that the plant is one connected graph from the roots
    to the leaves.

        Plant
         └─ Axis (ShootAxis) ─ GrowthUnit ─ Phytomer ─ Organ StemInternode (the subcrown internode)
             the seed (SubOrgan Seed), at the sowing depth: the seminal roots' bearer (ArchiSimple's root_anchor
             {"label": "Seed"})
             then subcrown_elements StemElements '<', vertical, up to the crown at crown_depth below the surface:
             the last one the crown's top, the main stem's bearer (Adel's shoot_anchor {"label": "StemElement",
             "last": True})

    Each element gets its start (x1, y1, z1), its end (x2, y2, z2), its length and its radius as MPG properties: the
    models anchored on it start at its end (the seed's: its position) and the other models read them (an anatomy, a
    hydraulic model). The seed is a point (its start and end at the seed), its length and radius the grain's. Its
    axis has no row in the shoot's axes table (axis_row -1): it is not the shoot's, nor its organ.

    The seed's depth is the planting's (z: minus the sowing depth, a planting_variable). The crown's depth does not follow the sowing depth (Kirby 1993): the subcrown
    internode's length is the sowing depth minus crown_depth, positive (a seed sown above the crown's depth raises).
    The initiator reads its parameters from the plant's scenario (the class defaults otherwise); it has no step.

    Its entities are its elements (the seed and the subcrown internode's StemElements: on its axis without a row,
    axis_row -1), at the plant's SubOrgans, so that it draws them as the shoot and the root models draw theirs
    (triangles(): the frames, the light): the subcrown internode's StemElements as cylinders, the seed as the grain,
    a cylinder of seed_length and seed_radius centred on the seed's position, lying along x. Without it the
    subcrown internode would not be drawn: the seminal roots, on the seed, would start a subcrown internode's length
    below the shoot, on the crown.
    """
    entities = {"label": ("Seed", "StemElement"), "axis_row": "<0"}

    # the planting (the seed's position)
    x: float = planting_variable("x")
    y: float = planting_variable("y")
    z: float = planting_variable("z")
    # the seed and the subcrown internode
    crown_depth: float = parameter(unit="m", description="The crown's depth below the soil surface (the tillering "
                                                         "node: the top of the subcrown internode).",
                                   by="WheatCrown", default=0.02, min_value=0.005, max_value=0.05,
                                   references=KIRBY_1993)
    seed_length: float = parameter(unit="m", description="The grain's length (the seed element's).", by="WheatCrown",
                                   default=0.0065, min_value=0.004, max_value=0.009, references=GEGAS_2010,
                                   DOI=GEGAS_2010_DOI)
    seed_radius: float = parameter(unit="m", description="Half the grain's width (the seed element's radius).",
                                   by="WheatCrown", default=0.0016, min_value=0.001, max_value=0.0025,
                                   references=GEGAS_2010, DOI=GEGAS_2010_DOI)
    subcrown_radius: float = parameter(unit="m", description="The subcrown internode's radius.", by="WheatCrown",
                                       default=0.0005, min_value=0.0002, max_value=0.0015,
                                       value_comment="without a source: a thin stem of about 1 mm across")
    subcrown_elements: float = parameter(unit="adim", description="The number of StemElements of the subcrown "
                                                                  "internode.", by="WheatCrown", default=2,
                                         min_value=1, max_value=20,
                                         value_comment="numerical: the internode's discretisation (e.g. for its "
                                                       "hydraulics)")
    # its entities: the elements of its axis, which has no row in the shoot's axes table
    axis_row: int = input_variable(unit="adim", description="The axis' row in the shoot's axes table (-1: not the "
                                                            "shoot's: the crown's axis).",
                                   by="model_shoot_architecture", default=-1, scale=ScalesConfig.Axis, dtype="int")
    # the subcrown internode's organ: its diameter, as the shoot model's organs have theirs (a stem anatomy's size)
    diameter: float = state_variable(unit="m", description="The subcrown internode's diameter (twice "
                                                           "subcrown_radius; the shoot's organs' diameter for the "
                                                           "others).", default=0., scale=ScalesConfig.Organ,
                                     kind="descriptor")

    @classmethod
    def plant_parameters(cls, scenario) -> dict:
        """The class's defaults overridden by *scenario*, the plant's."""
        return {f.name: scenario.get(f.name, f.default) for f in fields(cls)
                if f.metadata.get("variable_type") == "parameter"}

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        """The seed and the subcrown internode of *plant* (the class docstring's structure), from its planting's
        position and its *scenario*'s parameters."""
        parameters = cls.plant_parameters(scenario)
        x, y, seed_height = (float(g.property(name)[plant]) for name in ("x", "y", "z"))
        crown_height = float(getattr(g, "soil_surface", 0.)) - float(parameters["crown_depth"])   # below the soil
        if crown_height <= seed_height + 1e-12:
            raise ValueError(f"WheatCrown: the crown ({-crown_height} m deep) must lie above the seed "
                             f"({-seed_height} m deep): crown_depth below the sowing depth")
        count = int(parameters["subcrown_elements"])
        if count < 1 or count != parameters["subcrown_elements"]:
            raise ValueError(f"WheatCrown: subcrown_elements is a whole number >= 1, got "
                             f"{parameters['subcrown_elements']}")
        scales, labels = g.scales, g.labels
        complex_ = plant
        for scale, label in ((scales.Axis, labels.Axis.Shoot), (scales.GrowthUnit, labels.GrowthUnit.Shoot),
                             (scales.Phytomer, labels.Phytomer.Shoot), (scales.Organ, labels.Organ.StemInternode)):
            complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
        g.property("diameter")[complex_] = 2. * float(parameters["subcrown_radius"])
        element = g.add_component(complex_, **PropsConfig(scale=scales.SubOrgan, edge_type='/',
                                                         label=labels.SubOrgan.Seed))
        cls._place(g, element, x, y, seed_height, seed_height, float(parameters["seed_length"]),
                   float(parameters["seed_radius"]))
        rise = (crown_height - seed_height) / count
        for k in range(count):
            element = g.add_child(element, **PropsConfig(scale=scales.SubOrgan, edge_type='<',
                                                         label=labels.SubOrgan.StemElement))
            cls._place(g, element, x, y, seed_height + k * rise, seed_height + (k + 1) * rise, rise,
                       float(parameters["subcrown_radius"]))

    def triangles(self, where=None):
        """The elements' triangles (coupling.geometry.Triangles, each one's node index here): the subcrown internode's
        StemElements as cylinders from their start to their end, the seed as the grain (a cylinder of its length and
        its radius, centred on its position, along x); *where*: a node mask restricting them."""
        from openalea.metafspm.coupling.geometry import Triangles, segment_triangles
        ds = self.data_structure
        names = ("x1", "y1", "z1", "x2", "y2", "z2", "length", "radius")
        x1, y1, z1, x2, y2, z2, length, radius = (np.asarray(ds.get(name), dtype=float) for name in names)
        nodes = np.arange(x1.size) if where is None else np.flatnonzero(where)
        start = np.stack([x1[nodes], y1[nodes], z1[nodes]], axis=1)
        end = np.stack([x2[nodes], y2[nodes], z2[nodes]], axis=1)
        seed = np.asarray(ds.labels())[nodes] == "Seed"
        half_length = np.stack([0.5 * length[nodes][seed], np.zeros(seed.sum()), np.zeros(seed.sum())], axis=1)
        start[seed], end[seed] = start[seed] - half_length, start[seed] + half_length
        vertices, drawn = segment_triangles(start, end, radius[nodes])
        return Triangles(vertices[drawn].reshape(-1, 3, 3), np.repeat(nodes[drawn], vertices.shape[1]))

    @staticmethod
    def _place(g, element, x, y, bottom, top, length, radius):
        """An element's start and end (vertical, at x, y), its length and its radius, on the MPG."""
        for name, value in dict(x1=x, y1=y, z1=bottom, x2=x, y2=y, z2=top, length=length, radius=radius).items():
            g.property(name)[element] = value


# ═══════════════════════════════════════════════════════════════════════════════
# The nodal roots' emergence: NodalRootEmergence, a rule component
# ═══════════════════════════════════════════════════════════════════════════════

KLEPPER_1984 = ("Klepper B., Belford R.K., Rickman R.W. (1984) Root and shoot development in winter wheat. Agron. J. "
                "76: 117-122 (the nodal roots of the main stem's node n appear with its leaf n + 3, in pairs at the "
                "first nodes)")


@dataclass
class NodalRootEmergence(FunctionalComponent):
    """
    The nodal roots' emergence on a wheat plant, a rule from the shoot's development to the root model's sites:
    each leaf appeared on an axis asks for nodal_roots_per_leaf nodal roots on the phytomer nodal_root_rank_offset
    ranks below its own on that axis (its site: the root model grows them on the phytomer's bearer, its first
    SubOrgan, at the crown for the first phytomers), nodal_root_delay °Cd after the leaf's appearance (Klepper et
    al. 1984: the roots of node n with leaf n + 3). A leaf whose site would lie below the axis' first phytomer asks
    for none (the seminal roots are the seed's).

    The rule reads the shoot's public state (Adel's, ShootArchitecture's contract): each phytomer's rank
    (phytomer_rank) and axis (axis_row: the shoot's axes have a row in its axes table), each axis' Haun stage
    (haun_stage, the development's) and each plant's thermal time (thermal_time, the development's, °Cd). A leaf
    has appeared when its axis' Haun stage reaches its rank minus one (Haun 1973's scale: the leaf of rank n appears
    at stage n - 1); its appearance time is interpolated between the rule's last call and this one (the Haun stage
    linear in thermal time over a step), the call's thermal time for an axis the rule has not seen yet.

    It writes, per Phytomer, the cumulative number of nodal roots asked for on it (nodal_axes_requested, the root
    model's input: it grows the difference with those it grew), each leaf's request counted once when it falls due
    (leaf_nodal_roots): no request lost or replaced, whatever the steps; and each phytomer's node, where its nodal
    roots start (site_x, site_y, site_z, the root model's site position: _site_position), on the crown while the
    internodes below it have not elongated. Its entities are the shoot's elements (its axes with a row,
    axis_row >= 0), the phytomers they belong to its phytomers.
    """
    entities = {"axis_row": ">=0"}

    # read from the shoot (its architecture and its development)
    thermal_time: float = input_variable(unit="°Cd", description="The plant's thermal time (its development's).",
                                         by="model_development", default=np.nan, scale=ScalesConfig.Plant)
    haun_stage: float = input_variable(unit="adim", description="The axis' Haun stage: its leaves appeared, the last "
                                                                "one's fraction (its development's).",
                                       by="model_development", default=0., scale=ScalesConfig.Axis)
    axis_row: int = input_variable(unit="adim", description="The axis' row in the shoot's axes table (-1: not the "
                                                            "shoot's).", by="model_shoot_architecture", default=-1,
                                   scale=ScalesConfig.Axis, dtype="int")
    phytomer_rank: int = input_variable(unit="adim", description="The phytomer's rank on its axis, from 1.",
                                        by="model_shoot_architecture", default=0, scale=ScalesConfig.Phytomer,
                                        dtype="int")
    # the plant's position (its seed's) and the elements' lengths (the internodes', the shoot's and the crown's)
    x: float = planting_variable("x")
    y: float = planting_variable("y")
    z: float = planting_variable("z")
    length: float = input_variable(unit="m", description="The element's length (read on every element of the plant: "
                                                         "the internodes').", by="model_shoot_architecture",
                                   default=0., scale=ScalesConfig.SubOrgan)
    # the rule's state
    leaf_appearance_time: float = state_variable(unit="°Cd", description="The thermal time at which the phytomer's "
                                                                         "leaf appeared.",
                                                 value_comment="NaN: not appeared (or not a shoot's phytomer)",
                                                 kind="descriptor", default=np.nan, scale=ScalesConfig.Phytomer)
    leaf_nodal_roots: float = state_variable(unit="adim", description="The nodal roots the phytomer's leaf asked for "
                                                                      "on its site.",
                                             value_comment="NaN: not asked yet", kind="descriptor", default=np.nan,
                                             scale=ScalesConfig.Phytomer)
    site_x: float = state_variable(unit="m", description="Where the nodal roots of the phytomer start, x: its node "
                                                         "(the root model's site position).",
                                   value_comment="NaN: not a shoot's phytomer", kind="descriptor", default=np.nan,
                                   scale=ScalesConfig.Phytomer)
    site_y: float = state_variable(unit="m", description="Where the nodal roots of the phytomer start, y.",
                                   value_comment="NaN: not a shoot's phytomer", kind="descriptor", default=np.nan,
                                   scale=ScalesConfig.Phytomer)
    site_z: float = state_variable(unit="m", description="Where the nodal roots of the phytomer start, z: its node's "
                                                         "height, on the crown while the internodes below it have "
                                                         "not elongated.",
                                   value_comment="NaN: not a shoot's phytomer", kind="descriptor", default=np.nan,
                                   scale=ScalesConfig.Phytomer)
    nodal_axes_requested: float = state_variable(unit="adim", description="The cumulative number of nodal roots "
                                                                          "asked for on the phytomer (the root "
                                                                          "model's sites).", kind="descriptor",
                                                 default=0., scale=ScalesConfig.Phytomer)
    previous_thermal_time: float = state_variable(unit="°Cd", description="The axis' plant's thermal time at the "
                                                                          "rule's last call.",
                                                  value_comment="NaN: not seen yet", kind="descriptor",
                                                  default=np.nan, scale=ScalesConfig.Axis)
    previous_haun_stage: float = state_variable(unit="adim", description="The axis' Haun stage at the rule's last "
                                                                         "call.", value_comment="NaN: not seen yet",
                                                kind="descriptor", default=np.nan, scale=ScalesConfig.Axis)
    # the rule
    nodal_roots_per_leaf: float = parameter(unit="adim", description="The nodal roots each leaf appeared asks for.",
                                            by="NodalRootEmergence", default=2., min_value=0., max_value=6.,
                                            value_comment="two at the first nodes, more at the upper ones (Klepper "
                                                          "et al. 1984)", references=KLEPPER_1984)
    nodal_root_rank_offset: float = parameter(unit="adim", description="The ranks between a leaf and the phytomer "
                                                                       "bearing the nodal roots it asks for.",
                                              by="NodalRootEmergence", default=3., min_value=0., max_value=5.,
                                              references=KLEPPER_1984)
    nodal_root_delay: float = parameter(unit="°Cd", description="The thermal time from a leaf's appearance to its "
                                                                "nodal roots' emergence.", by="NodalRootEmergence",
                                        default=0., min_value=0., max_value=200.,
                                        value_comment="without a source: the phase is the rank offset's (Klepper "
                                                      "et al. 1984), the delay a refinement of it")

    @state
    def _nodal_axes_requested(self, thermal_time, haun_stage, axis_row, phytomer_rank, leaf_appearance_time,
                              leaf_nodal_roots, nodal_axes_requested, previous_thermal_time, previous_haun_stage,
                              nodal_roots_per_leaf, nodal_root_rank_offset, nodal_root_delay) -> dict:
        """
        Per Phytomer: the leaves appeared since the last call (their appearance times), the requests falling due
        (thermal time >= appearance + delay) added to their sites' cumulative counts, once each.
        """
        ds = self.data_structure
        axis = np.asarray(ds.owner("Axis", of="Phytomer"), dtype=np.int64)
        plant = np.asarray(ds.owner("Plant", of="Phytomer"), dtype=np.int64)
        rank = np.asarray(phytomer_rank, dtype=np.int64)
        own = np.zeros(rank.size, dtype=bool)
        own[np.unique(np.asarray(ds.owner("Phytomer"), dtype=np.int64))] = True     # its elements' phytomers
        own &= rank >= 1
        time = np.asarray(thermal_time, dtype=float)[plant]
        stage = np.asarray(haun_stage, dtype=float)[axis]
        time_before = np.asarray(previous_thermal_time, dtype=float)[axis]
        stage_before = np.asarray(previous_haun_stage, dtype=float)[axis]

        # The leaves appeared (the Haun stage at their rank minus one), their time interpolated over the step
        appearance = np.array(leaf_appearance_time, dtype=float)
        appeared = own & np.isnan(appearance) & np.isfinite(time) & (stage >= rank - 1)
        seen = appeared & np.isfinite(time_before) & np.isfinite(stage_before) & (stage > stage_before)
        fraction = np.zeros(rank.size)
        fraction[seen] = np.clip((rank[seen] - 1 - stage_before[seen]) / (stage[seen] - stage_before[seen]), 0., 1.)
        appearance[appeared] = np.where(seen, time_before + fraction * (time - time_before), time)[appeared]

        # The requests falling due, each added once to its site (the phytomer offset ranks below on its axis)
        asked = np.array(leaf_nodal_roots, dtype=float)
        requested = np.array(nodal_axes_requested, dtype=float)
        due = own & np.isnan(asked) & np.isfinite(appearance) & (time >= appearance + np.asarray(nodal_root_delay))
        site_rank = rank - np.round(np.asarray(nodal_root_rank_offset, dtype=float)).astype(np.int64)
        site = self._phytomer_at(axis, rank, own, site_rank)
        count = np.asarray(nodal_roots_per_leaf, dtype=float) * np.ones(rank.size)
        asked[due] = np.where(site[due] >= 0, count[due], 0.)
        given = due & (site >= 0)
        np.add.at(requested, site[given], count[given])
        return {"leaf_appearance_time": appearance, "leaf_nodal_roots": asked, "nodal_axes_requested": requested}

    @state
    def _site_position(self, x, y, z, phytomer_rank) -> dict:
        """
        Per Phytomer of the shoot: its node, where its nodal roots start (the root model's site_x, site_y, site_z):
        above the seed (the planting's position) by the internodes' lengths on its path from the seed, its own
        internode's included (the subcrown internode, then the shoot's: the elements of StemInternode organs but the
        seed, the grain, in the subcrown internode's organ), the axes vertical (Adel's tiller_inclination 0). The shoot model draws a hidden element on the top of the visible
        parts below it (Adel's turtle: a phytomer on its previous one's sheath), not at its node: a phytomer of the
        tillering zone, its internodes not elongated, has its node on the crown while its first element is drawn
        above the coleoptile, out of the soil.
        """
        ds = self.data_structure
        elements = ds.base                                   # every SubOrgan of the plant: the crown's, the roots'
        internode = ((np.asarray(elements.labels("Organ")) == "StemInternode")
                     & (np.asarray(elements.labels()) != "Seed"))
        own_length = np.where(internode, np.asarray(elements.get("length"), dtype=float), 0.)
        rise = elements.accumulate(own_length, direction="down")          # the path's internodes, its own included
        phytomer = np.asarray(elements.owner("Phytomer"), dtype=np.int64)
        count = len(elements.entity_ids("Phytomer"))
        kept = (phytomer >= 0) & internode
        height = np.full(count, -np.inf)
        np.maximum.at(height, phytomer[kept], rise[kept])               # the top of its own internode
        plant = np.asarray(elements.owner("Plant", of="Phytomer"), dtype=np.int64)
        shoot = (np.asarray(phytomer_rank) >= 1) & np.isfinite(height)
        site_x, site_y, site_z = np.full(count, np.nan), np.full(count, np.nan), np.full(count, np.nan)
        site_x[shoot] = np.asarray(x, dtype=float)[plant[shoot]]
        site_y[shoot] = np.asarray(y, dtype=float)[plant[shoot]]
        site_z[shoot] = np.asarray(z, dtype=float)[plant[shoot]] + height[shoot]
        return {"site_x": site_x, "site_y": site_y, "site_z": site_z}

    @state
    def _previous_axis_state(self, thermal_time, haun_stage) -> dict:
        """Per Axis: its plant's thermal time and its Haun stage at this call, for the next call's interpolation
        (after the requests)."""
        plant = np.asarray(self.data_structure.owner("Plant", of="Axis"), dtype=np.int64)
        return {"previous_thermal_time": np.asarray(thermal_time, dtype=float)[plant],
                "previous_haun_stage": np.array(haun_stage, dtype=float)}

    @staticmethod
    def _phytomer_at(axis, rank, own, wanted_rank) -> np.ndarray:
        """Per phytomer, the index of its axis' own phytomer of rank *wanted_rank* (-1: none)."""
        found = np.full(rank.size, -1, dtype=np.int64)
        candidates = np.flatnonzero(own)
        if candidates.size == 0:
            return found
        width = int(rank.max()) + 1                                    # (axis, rank) as one key
        key = axis * width + rank
        order = candidates[np.argsort(key[candidates], kind="stable")]
        wanted = axis * width + wanted_rank
        position = np.searchsorted(key[order], wanted).clip(0, order.size - 1)
        hit = (wanted_rank >= 1) & (key[order][position] == wanted)
        found[hit] = order[position][hit]
        return found
