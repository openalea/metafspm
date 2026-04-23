"""
Design sketch: graph-system decorator API for metafspm.

This file contains NO running code.  It is a readability-first description of
what modeler-facing component classes should look like once the decorator layer
is built.  The mathematics is the primary specification; the decorator syntax
must serve it, not the other way around.

Answers to the four open design questions
-----------------------------------------
Q1. Per-field balance tags:   @node_balance(field="xylem_pressure") — explicit
Q2. Boundary decorator:       @boundary_condition(location="node"|"edge",
                                                  kind="dirichlet"|"neumann")
Q3. Typed edge conductances:  builder sums fields tagged location="edge" whose
                               variable_type matches — modeler declares typed
                               conductances; builder assembles K_total per edge
Q4. Field location:           declare(..., location="node"|"edge") — new keyword

Mathematical conventions
------------------------
  p_x, p_ph  : node state vectors  (shape n_nodes)
  c           : node solute concentration  (shape n_nodes)
  q           : edge flux vector  (shape n_edges)
  K_e         : edge conductance scalar per edge
  B           : internal incidence matrix  (n_nodes × n_edges)
                B[i,e] = +1 if edge e leaves node i
                        = -1 if edge e enters node i
  B_b         : boundary incidence matrix  (n_nodes × n_boundary_ports)
  q_b         : boundary flux vector
  L = B diag(K) B^T  : weighted graph Laplacian
  sigma_soil  : radial soil–xylem conductance per node  [m³ s⁻¹ MPa⁻¹]
  sigma_xph   : radial xylem–phloem conductance per node
  p_soil      : soil water potential per node  (prescribed, boundary input)
  J_radial    : net radial solute input per node  (source/sink)
  s_ph        : phloem assimilate loading rate per node

EquationContext fields available in every decorated method
----------------------------------------------------------
  ctx.graph.incidence            : B  (CSC sparse)
  ctx.graph.boundary_incidence   : B_b
  ctx.node_unknowns[name]        : current iterate, shape (n_nodes,)
  ctx.edge_unknowns[name]        : current iterate, shape (n_edges,)
  ctx.node_fields[name].values   : non-unknown node arrays
  ctx.edge_fields[name].values   : non-unknown edge arrays
  ctx.boundary_ports             : tuple[BoundaryPort]
  ctx.boundary_values(kind=...)  : prescribed values at boundary ports
  ctx.parameters                 : dict of scalar / array parameters
  ctx.dt                         : time step (None for steady-state)
  ctx.previous_node_fields       : state at t_n (None for steady-state)
"""

from dataclasses import dataclass
from openalea.metafspm.component import Model, declare
# future imports — not yet implemented:
# from openalea.metafspm.graph_system_decorators import (
#     graph_system,
#     node_balance, edge_law,
#     boundary_condition,
#     graph_jacobian,
#     graph_output,
# )
import numpy as np
from scipy.sparse import diags


# ===========================================================================
# Use case 1 — Nitrogen axial transport along root axis segments
#              (transient, node + edge unknowns, linear system)
# ===========================================================================
#
# Biology: xylem sap moves axially and carries solutes; at each segment a
# radial source term brings solutes in from the apoplast or active loading.
# The apex is a zero-flux boundary; the collar has a prescribed concentration.
#
# Unknowns:   c   = concentration per segment node         [mol m⁻³]
#             q   = axial solute flux per edge              [mol s⁻¹]
#
# Node balance (transient, backward Euler):
#
#   C_i (c_i^{n+1} - c_i^n) / dt  +  (B q)_i  +  (B_b q_b)_i  -  J_radial_i  =  0
#
#   where  (B q)_i   = net axial efflux from node i (positive = leaving)
#          (B_b q_b) = axial flux into/out of boundary ports (collar, apex)
#          J_radial  = net radial solute input (uptake from apoplast, active loading)
#                      positive = source into xylem lumen
#
# Edge constitutive law:
#
#   q_e  =  K_axial_e * (B^T c)_e  =  K_axial_e * (c_tail - c_head)
#
# Boundaries:
#   Collar  (node, Dirichlet): c[collar] = c_collar_prescribed
#   Apex    (edge, Neumann  ): q[apex_edge] = 0  (zero-flux tip)
#
# Solver: linear_direct  (system is linear once J_radial is given)
# ---------------------------------------------------------------------------

# @graph_system(
#     node_unknowns=["concentration"],
#     edge_unknowns=["axial_flux"],
#     method="linear_direct",
#     schedule_as="axial",
# )
# @dataclass
class NitrogenAxialTransport(Model):

    concentration: float = declare(
        unit="mol m-3", unit_comment="solute concentration in xylem lumen",
        description="Xylem solute concentration per segment. Node unknown.",
        min_value=0., max_value=1e4, value_comment="", references="", DOI=[],
        variable_type="state_variable", by="NitrogenAxialTransport",
        state_variable_type="intensive", edit_by="dev", default=0.5,
        # location="node",
    )

    axial_flux: float = declare(
        unit="mol s-1", unit_comment="",
        description="Net axial solute flux across one inter-segment edge. Edge unknown.",
        min_value=-1., max_value=1., value_comment="", references="", DOI=[],
        variable_type="state_variable", by="NitrogenAxialTransport",
        state_variable_type="extensive", edit_by="dev", default=0.,
        # location="edge",
    )

    K_axial: float = declare(
        unit="m3 s-1", unit_comment="conductance times cross-section area",
        description="Axial conductance for solute transport on each inter-segment edge.",
        min_value=0., max_value=1., value_comment="", references="", DOI=[],
        variable_type="parameter", by="NitrogenAxialTransport",
        state_variable_type="intensive", edit_by="dev", default=1e-12,
        # location="edge",
    )

    volumetric_capacity: float = declare(
        unit="m3", unit_comment="effective storage volume of xylem lumen per segment",
        description="C_i: storage coefficient in the transient node balance.",
        min_value=0., max_value=1., value_comment="", references="", DOI=[],
        variable_type="parameter", by="NitrogenAxialTransport",
        state_variable_type="extensive", edit_by="dev", default=1e-9,
        # location="node",
    )

    radial_solute_input: float = declare(
        unit="mol s-1", unit_comment="net radial influx per segment",
        description=(
            "J_radial_i: net solute source into the xylem lumen from radial pathways "
            "(apoplastic diffusion, active loading). Positive = source, negative = sink. "
            "Provided as input from a coupled radial-transport model."
        ),
        min_value=-1., max_value=1., value_comment="", references="", DOI=[],
        variable_type="input", by="NitrogenRadialTransport",
        state_variable_type="extensive", edit_by="dev", default=0.,
        # location="node",
    )

    # @node_balance(field="concentration")
    def _concentration_balance(self, ctx) -> np.ndarray:
        """
        Transient node conservation of solute.

        R_node = C (c - c_old)/dt + B q + B_b q_b - J_radial = 0

        Each term:
          C (c - c_old)/dt  — storage: change in moles in the xylem lumen
          B q               — net axial efflux from internal edges
          B_b q_b           — net axial flux through boundary ports
                              (collar inlet, zero-flux apex)
          J_radial          — radial source from apoplast / active loading
        """
        c       = ctx.node_unknowns["concentration"]
        c_old   = ctx.previous_node_fields["concentration"]
        q       = ctx.edge_unknowns["axial_flux"]
        C       = ctx.node_fields["volumetric_capacity"].values
        J       = ctx.node_fields["radial_solute_input"].values
        B       = ctx.graph.incidence
        B_b     = ctx.graph.boundary_incidence
        q_b     = ctx.boundary_values()             # collar flux, apex flux
        return C * (c - c_old) / ctx.dt + B @ q + B_b @ q_b - J

    # @edge_law
    def _axial_transport_law(self, ctx) -> np.ndarray:
        """
        Linear constitutive law at each inter-segment edge.

        R_edge = q - K_axial * (B^T c)
               = q - K_axial * (c_tail - c_head)

        Ohm / Fick analogy: flux is proportional to the concentration drop.
        """
        c       = ctx.node_unknowns["concentration"]
        q       = ctx.edge_unknowns["axial_flux"]
        K       = ctx.edge_fields["K_axial"].values
        B       = ctx.graph.incidence
        return q - K * (B.T @ c)

    # @boundary_condition(location="node", kind="dirichlet", field="concentration")
    def _collar_dirichlet(self, ctx) -> np.ndarray:
        """
        Dirichlet constraint at the collar (root base).
        Replaces the node balance row for the collar node.

        R_bc = c[collar] - c_collar_prescribed = 0
        """
        c_prescribed = ctx.parameters["collar_concentration"]
        collar_idx   = ctx.graph.node_local_index(ctx.parameters["collar_node_id"])
        return ctx.node_unknowns["concentration"][collar_idx] - c_prescribed

    # @boundary_condition(location="edge", kind="neumann", field="axial_flux")
    def _apex_zero_flux(self, ctx) -> np.ndarray:
        """
        Neumann constraint at the apex tip: zero axial flux.
        Replaces the edge law row for the apex edge.

        R_bc = q[apex_edge] = 0
        """
        apex_edge_idx = ctx.parameters["apex_edge_local_idx"]
        return ctx.edge_unknowns["axial_flux"][apex_edge_idx:apex_edge_idx + 1]


# ===========================================================================
# Use case 2 — Water / Münch pressure-flow  (coupled node-only, Newton)
# ===========================================================================
#
# Biology: xylem carries water from roots to shoots driven by soil–leaf
# gradient.  Phloem circulates assimilates driven by pressure flow (Münch).
# The two vessels exchange water radially at each node.  Soil provides water
# into the xylem; leaf loading drives phloem.
#
# Unknowns:   p_x   = xylem water potential per node       [MPa]
#             p_ph  = phloem turgor pressure per node       [MPa]
#
# Xylem node balance (steady-state):
#
#   (L_x p_x)_i
#     + sigma_xph_i (p_x_i - p_ph_i)       ← radial exchange with phloem
#     - sigma_soil_i (p_soil_i - p_x_i)    ← radial uptake from soil
#                                             (negative sign: soil→xylem = inflow)
#   = 0
#
#   expanded:
#   (L_x + diag(sigma_xph) + diag(sigma_soil)) p_x
#     - diag(sigma_xph) p_ph
#     - diag(sigma_soil) p_soil
#   = 0
#
# Phloem node balance (steady-state):
#
#   (L_ph p_ph)_i
#     - sigma_xph_i (p_x_i - p_ph_i)       ← exchange with xylem (inflow here)
#     - s_ph_i                               ← osmotic loading from assimilates
#   = 0
#
#   expanded:
#   (L_ph + diag(sigma_xph)) p_ph
#     - diag(sigma_xph) p_x
#     - s_ph
#   = 0
#
# Boundaries:
#   Collar xylem  (node, Dirichlet): p_x[collar] = p_x_collar
#   Leaf phloem   (node, Dirichlet): p_ph[leaf]  = p_ph_leaf_loading
#
# p_soil is NOT an unknown; it is a prescribed node field (input from a
# soil-water model) used in the xylem balance as a driving term.
#
# Solver: newton  (with optional analytic Jacobian)
# ---------------------------------------------------------------------------

# @graph_system(
#     node_unknowns=["xylem_pressure", "phloem_pressure"],
#     edge_unknowns=[],
#     method="newton",
#     schedule_as="axial",
# )
# @dataclass
class WaterMunchTransport(Model):

    xylem_pressure: float = declare(
        unit="MPa", unit_comment="water potential in xylem vessels",
        description="Xylem water potential per segment node. Node unknown.",
        min_value=-5., max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="state_variable", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=-0.1,
        # location="node",
    )

    phloem_pressure: float = declare(
        unit="MPa", unit_comment="turgor pressure in phloem sieve tubes",
        description="Phloem turgor pressure per segment node. Node unknown.",
        min_value=-1., max_value=2., value_comment="", references="", DOI=[],
        variable_type="state_variable", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=0.8,
        # location="node",
    )

    K_xylem: float = declare(
        unit="m4 s-1 MPa-1", unit_comment="",
        description="Axial hydraulic conductance of xylem vessels per edge.",
        min_value=0., max_value=1., value_comment="", references="", DOI=[],
        variable_type="parameter", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=1e-10,
        # location="edge",
    )

    K_phloem: float = declare(
        unit="m4 s-1 MPa-1", unit_comment="",
        description="Axial hydraulic conductance of phloem sieve tubes per edge.",
        min_value=0., max_value=1., value_comment="", references="", DOI=[],
        variable_type="parameter", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=1e-11,
        # location="edge",
    )

    sigma_xph: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="radial, per segment",
        description=(
            "sigma_xph_i: radial membrane conductance between xylem and phloem "
            "per node. Drives water exchange between the two vessels."
        ),
        min_value=0., max_value=1., value_comment="", references="", DOI=[],
        variable_type="parameter", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=1e-13,
        # location="node",
    )

    sigma_soil: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="radial, per segment",
        description=(
            "sigma_soil_i: radial soil–root conductance per node. "
            "Drives water uptake from soil into xylem. "
            "Depends on root anatomy, soil moisture, and root–soil contact."
        ),
        min_value=0., max_value=1., value_comment="", references="", DOI=[],
        variable_type="parameter", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=1e-12,
        # location="node",
    )

    soil_water_potential: float = declare(
        unit="MPa", unit_comment="",
        description=(
            "p_soil_i: soil water potential at each root segment node. "
            "Prescribed input from a soil model. Drives xylem uptake."
        ),
        min_value=-5., max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="input", by="SoilWaterModel",
        state_variable_type="intensive", edit_by="dev", default=-0.05,
        # location="node",
    )

    phloem_assimilate_loading: float = declare(
        unit="MPa s-1", unit_comment="equivalent osmotic pressure loading rate",
        description=(
            "s_ph_i: osmotic source term from assimilate loading into phloem "
            "per node. Provided by a carbon model. Positive = loading."
        ),
        min_value=-1., max_value=1., value_comment="", references="", DOI=[],
        variable_type="input", by="CarbonModel",
        state_variable_type="intensive", edit_by="dev", default=0.,
        # location="node",
    )

    # @node_balance(field="xylem_pressure")
    def _xylem_balance(self, ctx) -> np.ndarray:
        """
        Steady-state xylem node water balance.

        R_x = L_x p_x
              + diag(sigma_xph) (p_x - p_ph)    [radial loss to phloem]
              - diag(sigma_soil) (p_soil - p_x)  [radial gain from soil]
            = 0

        Sign convention: positive residual contribution = water leaving the node.
          L_x p_x            — axial divergence (positive = net axial efflux)
          sigma_xph (p_x-p_ph) — positive when p_x > p_ph (water flows xylem→phloem)
          -sigma_soil (p_soil-p_x) — negative when p_soil > p_x (uptake from soil)

        Expanded linear form:
          (L_x + diag(sigma_xph + sigma_soil)) p_x
            - diag(sigma_xph) p_ph
            - diag(sigma_soil) p_soil  =  0
        """
        p_x     = ctx.node_unknowns["xylem_pressure"]
        p_ph    = ctx.node_unknowns["phloem_pressure"]
        p_soil  = ctx.node_fields["soil_water_potential"].values
        K_x     = ctx.edge_fields["K_xylem"].values
        sig_xph = ctx.node_fields["sigma_xph"].values
        sig_s   = ctx.node_fields["sigma_soil"].values
        B       = ctx.graph.incidence
        L_x     = B @ diags(K_x) @ B.T
        return (L_x @ p_x
                + sig_xph * (p_x - p_ph)
                - sig_s   * (p_soil - p_x))

    # @node_balance(field="phloem_pressure")
    def _phloem_balance(self, ctx) -> np.ndarray:
        """
        Steady-state phloem node water balance.

        R_ph = L_ph p_ph
               - diag(sigma_xph) (p_x - p_ph)   [radial gain from xylem]
               - s_ph                             [assimilate loading source]
             = 0

        Sign convention:
          L_ph p_ph            — axial divergence in phloem
          -sigma_xph (p_x-p_ph) — when p_x > p_ph water enters phloem (negative)
          -s_ph                 — assimilate loading increases osmotic pressure
                                  (acts as a water sink in pressure units)

        Expanded form:
          (L_ph + diag(sigma_xph)) p_ph - diag(sigma_xph) p_x - s_ph  =  0
        """
        p_x     = ctx.node_unknowns["xylem_pressure"]
        p_ph    = ctx.node_unknowns["phloem_pressure"]
        s_ph    = ctx.node_fields["phloem_assimilate_loading"].values
        K_ph    = ctx.edge_fields["K_phloem"].values
        sig_xph = ctx.node_fields["sigma_xph"].values
        B       = ctx.graph.incidence
        L_ph    = B @ diags(K_ph) @ B.T
        return (L_ph @ p_ph
                - sig_xph * (p_x - p_ph)
                - s_ph)

    # @boundary_condition(location="node", kind="dirichlet", field="xylem_pressure")
    def _collar_xylem_dirichlet(self, ctx) -> np.ndarray:
        """
        Prescribed xylem pressure at the collar (root–shoot junction).
        Replaces the xylem balance row for the collar node.

        R_bc = p_x[collar] - p_x_collar_prescribed = 0
        """
        p_prescribed = ctx.parameters["collar_xylem_pressure"]
        collar_idx   = ctx.graph.node_local_index(ctx.parameters["collar_node_id"])
        return ctx.node_unknowns["xylem_pressure"][collar_idx] - p_prescribed

    # @boundary_condition(location="node", kind="dirichlet", field="phloem_pressure")
    def _leaf_phloem_dirichlet(self, ctx) -> np.ndarray:
        """
        Prescribed phloem pressure at the leaf loading site.
        Replaces the phloem balance row for the leaf node.

        R_bc = p_ph[leaf] - p_ph_leaf_prescribed = 0
        """
        p_prescribed = ctx.parameters["leaf_phloem_pressure"]
        leaf_idx     = ctx.graph.node_local_index(ctx.parameters["leaf_node_id"])
        return ctx.node_unknowns["phloem_pressure"][leaf_idx] - p_prescribed

    # @graph_jacobian
    def _analytic_jacobian(self, ctx) -> np.ndarray:
        """
        Optional 2n × 2n analytic Jacobian of [R_x ; R_ph] w.r.t. [p_x ; p_ph].

        The full system stacks unknowns as x = [p_x | p_ph]:

          J = [ d(R_x)/d(p_x)    d(R_x)/d(p_ph)  ]
              [ d(R_ph)/d(p_x)   d(R_ph)/d(p_ph) ]

            = [ L_x + diag(sig_xph + sig_soil)   -diag(sig_xph)              ]
              [ -diag(sig_xph)                     L_ph + diag(sig_xph)       ]

        If this method is absent the builder uses finite-difference Jacobian.
        """
        K_x     = ctx.edge_fields["K_xylem"].values
        K_ph    = ctx.edge_fields["K_phloem"].values
        sig_xph = ctx.node_fields["sigma_xph"].values
        sig_s   = ctx.node_fields["sigma_soil"].values
        B       = ctx.graph.incidence
        L_x     = B @ diags(K_x)  @ B.T
        L_ph    = B @ diags(K_ph) @ B.T
        n       = ctx.graph.n_nodes
        J       = np.zeros((2 * n, 2 * n))
        J[:n, :n]   = (L_x  + diags(sig_xph + sig_s)).toarray()
        J[:n, n:]   = -diags(sig_xph).toarray()
        J[n:, :n]   = -diags(sig_xph).toarray()
        J[n:, n:]   = (L_ph + diags(sig_xph)).toarray()
        return J


# ===========================================================================
# Use case 3 — Mecha cross-sectional anatomy hydraulic network
#              (heterogeneous typed edges, linear, direct solve, flux output)
# ===========================================================================
#
# Biology: at the cross-section scale each root segment contains cortex,
# endodermis, and stele cells connected by three transport pathways:
#   - transmembrane: water crosses a lipid bilayer  (high resistance)
#   - symplastic: water flows through plasmodesmata  (medium resistance)
#   - apoplastic: water moves in the cell wall continuum (low resistance)
# Soil water enters at outer cortex nodes; xylem is the sink at stele nodes.
#
# Unknowns:  p = water potential per anatomy node  [MPa]
#
# Node balance (steady-state):
#
#   (L_het p)_i = b_i    for all free (non-boundary) nodes
#
#   where L_het = B diag(K_total) B^T
#         K_total_e = K_membrane_e + K_symplastic_e + K_apoplastic_e
#                     (builder sums typed edge conductances; unused types = 0)
#         b_i assembled from Dirichlet substitution at boundary nodes
#
# The builder detects method="linear_direct" and uses the matrix path
# (L p = b) directly — no Newton loop.
#
# Boundaries:
#   Outer cortex (node, Dirichlet): p[outer_wall] = p_soil_outer   (soil contact)
#   Inner stele  (node, Dirichlet): p[stele_node] = p_xylem_inner  (xylem lumen)
#
# Post-processing:
#   Edge flux recovered after the solve:
#     q_e = K_total_e * (B^T p)_e = K_total_e * (p_tail - p_head)
#
# Solver: linear_direct
# ---------------------------------------------------------------------------

# @graph_system(
#     node_unknowns=["water_potential"],
#     edge_unknowns=[],
#     method="linear_direct",
#     schedule_as="axial",
# )
# @dataclass
class MechaAnatomyHydraulics(Model):

    water_potential: float = declare(
        unit="MPa", unit_comment="",
        description="Water potential at each anatomy node. Node unknown.",
        min_value=-10., max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="state_variable", by="MechaAnatomyHydraulics",
        state_variable_type="intensive", edit_by="dev", default=-0.2,
        # location="node",
    )

    # Three typed conductances on anatomy edges.
    # The builder sums them into K_total per edge; edges where a type is absent
    # keep their declared default (0) and contribute nothing to K_total.

    K_membrane: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="per anatomy edge",
        description=(
            "Transmembrane hydraulic conductance. Non-zero only on edges of "
            "type e_type='transmembrane' (cell membrane crossing)."
        ),
        min_value=0., max_value=1., value_comment="", references="", DOI=[],
        variable_type="parameter", by="MechaAnatomyHydraulics",
        state_variable_type="intensive", edit_by="dev", default=0.,
        # location="edge",
    )

    K_symplastic: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="per anatomy edge",
        description=(
            "Symplastic (plasmodesmata) hydraulic conductance. Non-zero only "
            "on edges of type e_type='symplastic'."
        ),
        min_value=0., max_value=1., value_comment="", references="", DOI=[],
        variable_type="parameter", by="MechaAnatomyHydraulics",
        state_variable_type="intensive", edit_by="dev", default=0.,
        # location="edge",
    )

    K_apoplastic: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="per anatomy edge",
        description=(
            "Apoplastic (cell-wall) hydraulic conductance. Non-zero only on "
            "edges of type e_type='apoplastic'."
        ),
        min_value=0., max_value=1., value_comment="", references="", DOI=[],
        variable_type="parameter", by="MechaAnatomyHydraulics",
        state_variable_type="intensive", edit_by="dev", default=0.,
        # location="edge",
    )

    # Boundary inputs: declared as 'input' so the coupling dependency is explicit
    # and the field appears in the model's self-documentation just like in UC2.

    soil_water_potential: float = declare(
        unit="MPa", unit_comment="bulk soil water potential at root surface",
        description=(
            "Prescribed water potential at outer cortex boundary nodes. "
            "Provided by a soil-water model. The outer cortex cells are assumed "
            "in direct equilibrium with the soil solution at the anatomy scale."
        ),
        min_value=-10., max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="input", by="SoilWaterModel",
        state_variable_type="intensive", edit_by="dev", default=-0.05,
        # location="node",
    )

    xylem_water_potential: float = declare(
        unit="MPa", unit_comment="xylem lumen pressure at stele boundary",
        description=(
            "Prescribed water potential at inner stele boundary nodes. "
            "Provided by the organ-scale xylem solver (UC2). "
            "Couples the anatomy network to the axial transport system."
        ),
        min_value=-5., max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="input", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=-0.1,
        # location="node",
    )

    # @node_balance(field="water_potential")
    def _hydraulic_balance(self, ctx) -> np.ndarray:
        """
        Steady-state hydraulic node balance.

        R = L_het p - b = 0
            L_het = B diag(K_total) B^T
            K_total = K_membrane + K_symplastic + K_apoplastic  (summed by builder)
            b = boundary RHS assembled by Dirichlet substitution

        For method="linear_direct" the builder extracts the matrix L_het
        and vector b from this residual without Newton iteration.
        The modeler writes the residual form; the builder decides the solve path.
        """
        p       = ctx.node_unknowns["water_potential"]
        K_total = (ctx.edge_fields["K_membrane"].values
                   + ctx.edge_fields["K_symplastic"].values
                   + ctx.edge_fields["K_apoplastic"].values)
        b       = ctx.parameters["boundary_rhs"]   # assembled from Dirichlet BCs
        B       = ctx.graph.incidence
        L_het   = B @ diags(K_total) @ B.T
        return L_het @ p - b

    # @boundary_condition(location="node", kind="dirichlet", field="water_potential")
    def _outer_cortex_dirichlet(self, ctx) -> np.ndarray:
        """
        Soil water potential imposed at the outer cortex wall nodes.
        These nodes are in direct contact with the soil solution.

        R_bc = p[outer_wall_nodes] - p_soil = 0

        p_soil comes from the declared 'soil_water_potential' input field,
        not from an opaque parameters dict — so the coupling is explicit and
        visible in the model's field documentation.
        """
        outer_ids = ctx.parameters["outer_cortex_node_ids"]   # node index mask, structural
        p_soil    = ctx.node_fields["soil_water_potential"].values[outer_ids]
        return ctx.node_unknowns["water_potential"][outer_ids] - p_soil

    # @boundary_condition(location="node", kind="dirichlet", field="water_potential")
    def _inner_stele_dirichlet(self, ctx) -> np.ndarray:
        """
        Xylem lumen pressure imposed at the inner stele boundary nodes.
        Couples the anatomy-scale network to the organ-scale xylem solver (UC2).

        R_bc = p[stele_nodes] - p_xylem = 0
        """
        stele_ids = ctx.parameters["stele_node_ids"]           # node index mask, structural
        p_xylem   = ctx.node_fields["xylem_water_potential"].values[stele_ids]
        return ctx.node_unknowns["water_potential"][stele_ids] - p_xylem

    # @graph_output(name="edge_water_flux")
    def _derive_edge_fluxes(self, ctx) -> np.ndarray:
        """
        Post-solve: recover water flux along each anatomy edge.

        q_e = K_total_e * (B^T p)_e  =  K_total_e * (p_tail - p_head)

        This is the anatomical equivalent of Mecha's _calculate_edge_fluxes.
        Computed after the linear solve; not part of the system residual.
        """
        p       = ctx.node_unknowns["water_potential"]
        K_total = (ctx.edge_fields["K_membrane"].values
                   + ctx.edge_fields["K_symplastic"].values
                   + ctx.edge_fields["K_apoplastic"].values)
        B       = ctx.graph.incidence
        return K_total * (B.T @ p)
