"""
LegacyMPGDataStructure — illustrative example.

Demonstrates that plotting functions operating exclusively through the
GraphDataStructure abstract API (node_ids, edges, node_property,
incidence_matrix) require zero changes when the concrete implementation
is LegacyMPGDataStructure.

Plant scenario
--------------
A two-phytomer wheat seedling with one ramified root axis.  Water potential (ψ)
is assigned to reflect a mild evaporative gradient:
  shoot collar  ψ ≈ −0.2 MPa  (well supplied from root)
  leaf tips     ψ ≈ −1.3 MPa  (transpiring)
  root tips     ψ ≈ −0.3 MPa  (absorbing from soil)

The data is stored directly in g.property('water_potential') — the native
OpenAlea dict format — then wrapped in LegacyMPGDataStructure.
All downstream consumers (plot_graph_property, plot_incidence_structure)
never touch g.property(); they only call the abstract API.

Output
------
example_legacy_mpg.png  (saved next to this file)
"""

import sys
import os
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'mpg_tests'))
sys.path.insert(0, _HERE)   # examples/plotting takes priority over mpg_tests/plotting

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openalea.metafspm.data_structure.data_api import LegacyMPGDataStructure
from simple_seedling import generate_simple_mpg_seedling
from plotting import plot_graph_property, plot_incidence_structure


# ── 1. Build plant topology ───────────────────────────────────────────────────

g, sd = generate_simple_mpg_seedling()

# ── 2. Assign water potential in the native MTG dict format ──────────────────
#       (concrete-class setup — everything below uses abstract API only)

wp = g.property('water_potential')
# shoot
wp[sd.internodeelement]  = -0.20   # stem base (collar)
wp[sd.internodeelement2] = -0.40   # second internode
wp[sd.leafelement1]      = -0.75   # leaf 1 base
wp[sd.leafelement2]      = -1.00
wp[sd.leafelement3]      = -1.20   # leaf 1 tip
wp[sd.leafelement4]      = -0.85   # leaf 2 base
wp[sd.leafelement5]      = -1.10
wp[sd.leafelement6]      = -1.30   # leaf 2 tip
# root
wp[sd.root_segment1]     = -0.10   # root collar
wp[sd.root_segment2]     = -0.15
wp[sd.root_segment3]     = -0.22   # primary root tip
wp[sd.root_segment4]     = -0.18   # lateral root base
wp[sd.root_segment5]     = -0.25
wp[sd.root_segment6]     = -0.30   # lateral root tip

# ── 3. Wrap in LegacyMPGDataStructure ─────────────────────────────────────────

ds: LegacyMPGDataStructure = LegacyMPGDataStructure(g, g.scales.SubOrgan)
# From here on, only the abstract GraphDataStructure API is used.

print(f"n_nodes : {ds.n_nodes()}")          # abstract API
print(f"n_edges : {ds.n_edges()}")          # abstract API
print(f"n_dof   : {ds.n_dof}")              # abstract API

# extract_state packs ψ into a flat solver-ready vector
psi_vec = ds.extract_state(['water_potential'])   # abstract API
print(f"ψ vector (n={psi_vec.size}): min={psi_vec.min():.2f}  max={psi_vec.max():.2f} MPa")

# ── 4. Plot using abstract API only ──────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "LegacyMPGDataStructure  —  dict-backed plant graph  (Level 4a)\n"
    "plot_graph_property and plot_incidence_structure use only the abstract API",
    fontsize=10,
)

# Left: node colours = water potential
plot_graph_property(
    ds, 'water_potential',
    title="Water potential ψ (MPa)\nanchor node = 0 (grey)",
    cmap='RdYlGn',
    ax=axes[0],
)

# Right: incidence matrix B  (dense — LegacyMPGDataStructure uses np.ndarray)
plot_incidence_structure(
    ds,
    title="Incidence matrix B  (dense)\n+1 = inflow node,  −1 = outflow node",
    ax=axes[1],
)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'example_legacy_mpg.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"Saved → {out}")
