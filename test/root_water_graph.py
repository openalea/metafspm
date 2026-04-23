"""
Reimplementation of RootWaterModel.water_transport_munch_arrays via the
graph-system decorator API.

Two architectural gaps in the original @graph_system decorator are addressed:

GAP 1 — Choregrapher lifecycle  [FIXED in graph_system_decorators.py]
    @graph_system now calls Choregrapher().add_process() at class-decoration
    time, creating a no-arg _graph_solve trampoline via types.FunctionType so
    that the function carries the decorated class's module globals (needed for
    the Choregrapher's `inheriting` lookup).  Because no args besides `self`
    means iterating=True, the Choregrapher calls fun(instance) directly —
    the correct hook for whole-graph solves that cannot be decomposed per vertex.

GAP 2 — BoundaryPort multi-unknown limitation
    GraphView.boundary_incidence is (n_nodes × n_ports) with no field
    discriminator.  Routing two collar BCs (xylem and phloem) through the same
    matrix column is ambiguous for a 2-unknown system.  Workaround: pass collar
    values as ctx.parameters entries rather than BoundaryPorts.
"""

import os
import sys
import numpy as np
from dataclasses import dataclass
from scipy.sparse import coo_matrix, csc_matrix

from openalea.metafspm.component import Model, declare
from openalea.metafspm.graph_system import FieldState, GraphView, EquationContext
from openalea.metafspm.graph_system_decorators import (
    graph_system, node_balance, graph_jacobian, graph_output,
)

sys.path.insert(0, os.path.dirname(__file__))
from root_water import RootWaterModel


# ── Choregrapher bridge ───────────────────────────────────────────────────────

@dataclass
class GraphModel(Model):
    """
    Mixin that provides _run_graph_system as the lifecycle hook for subclasses.

    @graph_system (not this class) handles Choregrapher registration: it
    injects a no-arg _graph_solve trampoline under the decorated class's name,
    which makes Functor.iterating=True and gets called directly by the
    Choregrapher — the correct hook for whole-graph solves.
    """

    def _run_graph_system(self):
        raise NotImplementedError


# ── Münch xylem + phloem transport via graph-system API ──────────────────────

@graph_system(
    node_unknowns=["xylem_pressure_in", "phloem_pressure_in"],
    edge_unknowns=[],
    method="newton",
    max_iter=2,    # linear physics → Newton converges in exactly 1 step
    tol=1e-12,
)
@dataclass
class RootWaterModelGraph(RootWaterModel, GraphModel):
    """
    Münch xylem + phloem water transport expressed via the graph-system API.

    Topology: directed tree (child → parent edges, one per non-root segment).
    Unknowns: xylem_pressure_in and phloem_pressure_in, packed as [P_x | P_ph].
    The analytic Jacobian is exact (linear physics), so Newton needs 1 step.

    GAP 1 resolved: @graph_system registers _graph_solve under "RootWaterModelGraph".
    GAP 2 resolved: collar BCs passed via ctx.parameters (no BoundaryPort).

    TODO: flux-BC collar branch (water_root_shoot_xylem / sucrose_root_to_shoot_phloem)
    TODO: topology rebuild each timestep for growing architectures
    TODO: skip support nodes (type_Support_for_*) in _build_graph_view
    """

    def __init__(self, g, time_step, **scenario):
        super().__init__(g, time_step, **scenario)
        self._graph_view = None

    def post_coupling_init(self):
        super().post_coupling_init()
        self._graph_view = self._build_graph_view()

    # ------------------------------------------------------------------
    # Graph topology builder
    # ------------------------------------------------------------------

    def _build_graph_view(self):
        """Build a compact GraphView from the current focus-element tree."""
        focus_vids = np.asarray(sorted(self.props["focus_elements"]), dtype=np.int64)
        n = focus_vids.size
        vid_to_local = {int(v): i for i, v in enumerate(focus_vids)}

        tail_list, head_list = [], []
        for i, v in enumerate(focus_vids):
            parent_v = self.g.parent(int(v))  # None for topological root
            if parent_v is not None and int(parent_v) in vid_to_local:
                tail_list.append(i)
                head_list.append(vid_to_local[int(parent_v)])

        tail = np.asarray(tail_list, dtype=np.int64)
        head = np.asarray(head_list, dtype=np.int64)
        m = tail.size
        e = np.arange(m, dtype=np.int64)

        if m > 0:
            incidence = coo_matrix(
                (np.r_[np.ones(m), -np.ones(m)],
                 (np.r_[tail, head], np.r_[e, e])),
                shape=(n, m),
            ).tocsc()
        else:
            incidence = csc_matrix((n, 0), dtype=np.float64)

        return GraphView(
            node_ids=focus_vids,
            edge_ids=e,
            tail=tail,
            head=head,
            incidence=incidence,
            boundary_incidence=csc_matrix((n, 0), dtype=np.float64),  # GAP 2
            boundary_names=(),
        )

    # ------------------------------------------------------------------
    # Props ↔ FieldState synchronisation
    # ------------------------------------------------------------------

    def _sync_fields_from_props(self):
        """Snapshot current props into FieldState + parameters dict."""
        props = self.props
        vids = self._graph_view.node_ids  # sorted focus vids

        def _pull(name):
            return np.asarray([props[name][int(v)] for v in vids], dtype=np.float64)

        def _volumic(name_c, name_m, name_v):
            vol = _pull(name_v)
            safe = np.where(vol > 0, vol, 1.)
            return np.where(vol > 0, _pull(name_c) * _pull(name_m) / safe, 0.)

        gv = self._graph_view
        node_fields = {
            "xylem_pressure_in":  FieldState("xylem_pressure_in",  "node", _pull("xylem_pressure_in")),
            "phloem_pressure_in": FieldState("phloem_pressure_in", "node", _pull("phloem_pressure_in")),
        }
        parameters = {
            "K_xylem":      _pull("K_xylem"),
            "K_phloem":     _pull("K_phloem"),
            "kr_x":         _pull("kr_symplasmic_water_xylem") + _pull("kr_apoplastic_water_xylem"),
            "kr_ph":        _pull("kr_symplasmic_water_phloem"),
            "P_soil":       _pull("soil_water_pressure"),
            "RT":           8.31415 * (273.15 + _pull("soil_temperature")),
            "Cv_soil":      _pull("Cv_solutes_soil"),
            "Cv_x":         _volumic("C_solutes_xylem",  "living_struct_mass", "xylem_volume"),
            "Cv_ph":        _volumic("C_solutes_phloem", "living_struct_mass", "phloem_volume"),
            "sigma_x":      self.reflection_xylem,
            "sigma_ph":     self.reflection_phloem,
            # collar BCs — GAP 2 workaround (no BoundaryPort field discriminator)
            "p_x_collar":   float(props["xylem_pressure_collar"][1]),
            "p_ph_collar":  float(props["phloem_pressure_collar"][1]),
            "collar_local": int(gv.node_local_index(1)),
            # tree topology aligned to local indices
            "children":     gv.tail,   # edge tail = child local idx
            "parents":      gv.head,   # edge head = parent local idx
        }
        return node_fields, {}, parameters

    def _write_back_to_props(self, packed, outputs):
        vids = self._graph_view.node_ids
        n = vids.size
        props = self.props
        for i, v in enumerate(vids):
            v = int(v)
            props["xylem_pressure_in"][v]  = packed[i]
            props["phloem_pressure_in"][v] = packed[n + i]
        for name, arr in outputs.items():
            for i, v in enumerate(self._graph_view.node_ids):
                props[name][int(v)] = arr[i]

    # ------------------------------------------------------------------
    # Lifecycle hook called by GraphModel._graph_solve
    # ------------------------------------------------------------------

    def _run_graph_system(self):
        node_fields, edge_fields, params = self._sync_fields_from_props()
        system = self.build_graph_system(
            graph=self._graph_view,
            node_fields=node_fields,
            edge_fields=edge_fields,
            parameters=params,
        )
        packed = system.solve()
        self._write_back_to_props(packed, system.derive_outputs(packed))

    # ------------------------------------------------------------------
    # Residual blocks
    # ------------------------------------------------------------------

    @node_balance(field="xylem_pressure_in")
    def _xylem_residual(self, ctx: EquationContext) -> np.ndarray:
        p    = ctx.parameters
        P_x  = ctx.node_unknowns["xylem_pressure_in"]
        P_ph = ctx.node_unknowns["phloem_pressure_in"]
        ch, pa = p["children"], p["parents"]
        n = P_x.size

        # Parent pressure: tree coupling for non-root; fixed BC at collar
        P_x_par = np.empty(n)
        P_x_par[ch] = P_x[pa]
        P_x_par[p["collar_local"]] = p["p_x_collar"]

        osmotic_x  = p["sigma_x"]  * p["RT"] * (p["Cv_soil"] - p["Cv_x"])
        osmotic_ph = p["sigma_ph"] * p["RT"] * (p["Cv_ph"]   - p["Cv_x"])

        return (
            p["K_xylem"] * (P_x - P_x_par)
            - np.bincount(pa, weights=p["K_xylem"][ch] * (P_x[ch] - P_x[pa]), minlength=n)
            - p["kr_x"]  * (p["P_soil"] - P_x - osmotic_x)
            - p["kr_ph"] * (P_ph - P_x - osmotic_ph)
        )

    @node_balance(field="phloem_pressure_in")
    def _phloem_residual(self, ctx: EquationContext) -> np.ndarray:
        p    = ctx.parameters
        P_x  = ctx.node_unknowns["xylem_pressure_in"]
        P_ph = ctx.node_unknowns["phloem_pressure_in"]
        ch, pa = p["children"], p["parents"]
        n = P_ph.size

        P_ph_par = np.empty(n)
        P_ph_par[ch] = P_ph[pa]
        P_ph_par[p["collar_local"]] = p["p_ph_collar"]

        osmotic_ph = p["sigma_ph"] * p["RT"] * (p["Cv_ph"] - p["Cv_x"])

        return (
            p["K_phloem"] * (P_ph - P_ph_par)
            - np.bincount(pa, weights=p["K_phloem"][ch] * (P_ph[ch] - P_ph[pa]), minlength=n)
            + p["kr_ph"] * (P_ph - P_x - osmotic_ph)
        )

    # ------------------------------------------------------------------
    # Analytic Jacobian — 2n×2n (linear physics: exact in 1 Newton step)
    # ------------------------------------------------------------------

    @graph_jacobian
    def _jacobian(self, ctx: EquationContext) -> np.ndarray:
        p = ctx.parameters
        n = ctx.node_unknowns["xylem_pressure_in"].size
        ch, pa = p["children"], p["parents"]
        K_x, K_ph = p["K_xylem"], p["K_phloem"]
        kr_x, kr_ph = p["kr_x"], p["kr_ph"]

        sum_K_ch_x  = np.bincount(pa, weights=K_x[ch],  minlength=n)
        sum_K_ch_ph = np.bincount(pa, weights=K_ph[ch], minlength=n)
        i = np.arange(n, dtype=np.int64)

        # Sparsity pattern in [P_x | P_ph] layout; 4n + 4(n-1) = 8n-4 non-zeros
        rows = np.concatenate([
            i,       i + n,   # diagonals
            i,       i + n,   # cross terms dG_x/dP_ph, dG_ph/dP_x
            ch,      ch + n,  # child row → parent col  (axial coupling up)
            pa,      pa + n,  # parent row → child col  (axial coupling down)
        ])
        cols = np.concatenate([
            i,       i + n,
            i + n,   i,
            pa,      pa + n,
            ch,      ch + n,
        ])
        vals = np.concatenate([
            K_x + sum_K_ch_x + kr_x + kr_ph,   # dG_x/dP_x[i]
            K_ph + sum_K_ch_ph + kr_ph,          # dG_ph/dP_ph[i]
            -kr_ph,  -kr_ph,                     # cross derivatives
            -K_x[ch],  -K_ph[ch],               # parent coupling
            -K_x[ch],  -K_ph[ch],               # children coupling
        ])
        return csc_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n)).toarray()

    # ------------------------------------------------------------------
    # Post-solve output hooks
    # ------------------------------------------------------------------

    @graph_output(name="axial_export_water_up_xylem")
    def _axial_xylem(self, ctx: EquationContext) -> np.ndarray:
        p = ctx.parameters
        P_x = ctx.node_unknowns["xylem_pressure_in"]
        P_x_par = np.empty(P_x.size)
        P_x_par[p["children"]] = P_x[p["parents"]]
        P_x_par[p["collar_local"]] = p["p_x_collar"]
        return p["K_xylem"] * (P_x - P_x_par)

    @graph_output(name="axial_export_water_up_phloem")
    def _axial_phloem(self, ctx: EquationContext) -> np.ndarray:
        p = ctx.parameters
        P_ph = ctx.node_unknowns["phloem_pressure_in"]
        P_ph_par = np.empty(P_ph.size)
        P_ph_par[p["children"]] = P_ph[p["parents"]]
        P_ph_par[p["collar_local"]] = p["p_ph_collar"]
        return p["K_phloem"] * (P_ph - P_ph_par)
