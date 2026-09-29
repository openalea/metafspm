"""
MPGDataStructure — illustrative example.

Demonstrates three things:

1. ``from_legacy`` migration: a LegacyMPGDataStructure (dict-backed) is
   converted to an MPGDataStructure (numpy-backed) in one call.  The same
   plotting function is called on both instances and produces the same picture.

2. Sparse incidence matrix: ``ds.incidence_matrix()`` now returns a CSR sparse
   matrix.  The same ``plot_incidence_structure`` function handles both dense
   and sparse transparently.  The sparse B enables efficient graph Laplacian
   assembly: L = B K B^T — the key operation in every transport solver.

3. Solver-like state update via ``inject_state``: without touching any
   class-specific attribute the caller can push a new ψ vector into the data
   structure, exactly as the GraphSystemBuilder will do after each Newton step.

Plant scenario
--------------
Same two-phytomer seedling.  ψ values are loaded from the legacy format, then
the numpy-backed structure runs one explicit relaxation step toward a uniform
potential field, demonstrating the round-trip  extract_state → compute → inject_state.

Output
------
example_mpg_data_structure.png  (saved next to this file)
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
from scipy.sparse import diags, issparse

from openalea.metafspm.data_structure.data_api import (
    LegacyMPGDataStructure,
    MPGDataStructure,
)
from simple_seedling import generate_simple_mpg_seedling
from plotting import plot_graph_property, plot_incidence_structure


# ── 1. Build legacy data structure (concrete setup) ───────────────────────────

g, sd = generate_simple_mpg_seedling()

wp = g.property('water_potential')
wp[sd.internodeelement]  = -0.20;  wp[sd.internodeelement2] = -0.40
wp[sd.leafelement1]      = -0.75;  wp[sd.leafelement2]      = -1.00
wp[sd.leafelement3]      = -1.20;  wp[sd.leafelement4]      = -0.85
wp[sd.leafelement5]      = -1.10;  wp[sd.leafelement6]      = -1.30
wp[sd.root_segment1]     = -0.10;  wp[sd.root_segment2]     = -0.15
wp[sd.root_segment3]     = -0.22;  wp[sd.root_segment4]     = -0.18
wp[sd.root_segment5]     = -0.25;  wp[sd.root_segment6]     = -0.30

ds_leg: LegacyMPGDataStructure = LegacyMPGDataStructure(g, g.scales.SubOrgan)

# ── 2. Migrate to numpy-backed MPGDataStructure ────────────────────────────────
#       One call — copies specified properties as contiguous numpy arrays.

ds_np: MPGDataStructure = MPGDataStructure.from_legacy(ds_leg, ['water_potential'])
print(f"Migrated  n_nodes={ds_np.n_nodes()}  n_edges={ds_np.n_edges()}")

# Verify: extract_state gives the same vector from both implementations
psi_leg = ds_leg.extract_state(['water_potential'])   # abstract API
psi_np  = ds_np.extract_state(['water_potential'])    # abstract API
assert np.allclose(psi_leg, psi_np), "Migration must preserve values exactly"
print("extract_state: Legacy ≡ MPGDataStructure ✓")

# ── 3. Sparse graph Laplacian — enabled by sparse incidence matrix ─────────────

B = ds_np.incidence_matrix()    # abstract API → sparse CSR  (MPGDataStructure)
print(f"B is sparse: {issparse(B)}  shape: {B.shape}")

# Uniform conductances for illustration
m = ds_np.n_edges()
K = diags(np.ones(m), format='csr')   # K diagonal = conductance per edge
L_graph = B @ K @ B.T                 # graph Laplacian: (n×m)(m×m)(m×n) → (n×n) sparse
print(f"Graph Laplacian L: shape {L_graph.shape}  nnz={L_graph.nnz}")

# ── 4. Solver-like relaxation step via abstract API only ──────────────────────
#       Emulates one Newton-like update: ψ ← ψ − α · L · ψ

alpha = 0.05
psi   = ds_np.extract_state(['water_potential'])   # abstract API
psi_new = psi - alpha * (L_graph @ psi)
ds_np.inject_state(psi_new, ['water_potential'])   # abstract API
print(f"After relaxation: ψ range [{psi_new.min():.3f}, {psi_new.max():.3f}] MPa")

# ── 5. Plot — same functions, both implementations ────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    "MPGDataStructure  —  from_legacy migration + sparse B  (Level 4b)\n"
    "Same plotting API, same picture; B is now CSR sparse for efficient  L = B K Bᵀ",
    fontsize=10,
)

# Left: legacy ψ  (baseline before relaxation — re-read from legacy ds)
plot_graph_property(
    ds_leg, 'water_potential',
    title="Legacy  ψ (MPa)\n(dict-backed, Level 4a)",
    cmap='RdYlGn', ax=axes[0],
)

# Middle: numpy ψ after one relaxation step
plot_graph_property(
    ds_np, 'water_potential',
    title="MPGDataStructure  ψ after relaxation\n(numpy-backed, Level 4b)",
    cmap='RdYlGn', ax=axes[1],
)

# Right: sparse B — same plot_incidence_structure call as for legacy
plot_incidence_structure(
    ds_np,
    title="Incidence matrix B  (sparse CSR)\nfoundation for L = B K Bᵀ",
    ax=axes[2],
)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'example_mpg_data_structure.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"Saved → {out}")
