"""
MultiGridDataStructure — illustrative example.

Demonstrates that:
  - ``MultiGridDataStructure`` delegates the entire FieldDataStructure API
    to its finest grid, so plot_1d_profile and diffusion_step work unchanged.
  - ``restrict`` / ``prolongate`` explicitly expose what multigrid methods
    exploit: coarse grids capture low-frequency content; fine-scale detail
    is the residual that only the fine grid can represent.

Physical scenario
-----------------
A 1-D light-extinction profile in a leaf canopy, 64 nodes over 2 m height.
Signal: a broad Gaussian illumination peak (low frequency) superimposed with
fine-scale leaf-clumping modulation (high frequency).

Three-level hierarchy is built by ``from_coarsening(fine, n_levels=3, factor=2)``:
  level 0  →  64 nodes (fine  Δz = 0.03125 m)
  level 1  →  32 nodes
  level 2  →  16 nodes (coarse Δz = 0.125 m)

Panel A — multiresolution view
    The same signal restricted to each grid level, plotted together.
    Low-frequency peak is preserved; fine clumping disappears at coarser levels.

Panel B — restriction–prolongation error
    error(z) = signal(z) − P(R(signal))(z)
    Shows exactly the sub-grid detail that is invisible to a coarse-grid solver:
    this is what a multigrid correction step must approximate.

Output
------
example_multigrid_data_structure.png  (saved next to this file)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openalea.metafspm.data_structure.data_api import (
    ArrayDataStructure,
    MultiGridDataStructure,
    FieldDataStructure,
)
from plotting import plot_1d_profile, plot_multigrid_levels


# ── Generic diffusion integrator (identical to array example) ─────────────────

def diffusion_step(
    ds: FieldDataStructure, var_name: str, D: float, dt: float, n_steps: int = 1
) -> None:
    """Abstract API only: laplacian, extract_state, inject_state."""
    L = ds.laplacian()
    for _ in range(n_steps):
        u = ds.extract_state([var_name])
        ds.inject_state(u + dt * D * (L @ u), [var_name])


# ── 1. Build the multi-level hierarchy ───────────────────────────────────────

n_fine = 64
height = 2.0          # m  (canopy height)
dz     = height / n_fine

fine = ArrayDataStructure(shape=(n_fine,), dx=dz, origin=np.array([0.0]))

# Canopy light signal: broad Gaussian + fine-scale clumping oscillation
z_fine   = fine.coordinates()[:, 0]
signal   = (np.exp(-((z_fine - 1.2) ** 2) / (2 * 0.25 ** 2))   # Gaussian peak
            + 0.15 * np.sin(16 * np.pi * z_fine / height))       # leaf clumping

fine.add_field('irradiance', signal)

mg = MultiGridDataStructure.from_coarsening(fine, n_levels=3, factor=2)

print(f"Fine  n_dof = {mg.n_dof}")       # abstract API — delegates to fine
print(f"n_levels    = {mg.n_levels}")
for lvl in range(mg.n_levels):
    print(f"  level {lvl}: n = {mg._levels[lvl].grid.n_dof}")

# ── 2. Restriction–prolongation error ────────────────────────────────────────
#       All operations via abstract API

sig_fine  = mg.extract_state(['irradiance'])          # abstract API
sig_lv1   = mg.restrict(sig_fine, from_level=0)       # abstract API
sig_coarse = mg.restrict(sig_lv1,  from_level=1)      # abstract API
sig_back_lv1 = mg.prolongate(sig_coarse, to_level=1)  # abstract API
sig_back_fine = mg.prolongate(sig_back_lv1, to_level=0) # abstract API
error     = sig_fine - sig_back_fine

print(f"RP error  max|e| = {np.abs(error).max():.4f}  "
      f"RMS = {np.sqrt((error**2).mean()):.4f}")

# ── 3. Apply diffusion through the abstract FieldDataStructure interface ──────
#       mg acts as a FieldDataStructure — diffusion_step doesn't know about levels

D_canopy = 0.02                               # effective turbulent diffusivity (m² s⁻¹)
dt       = 0.4 * dz**2 / D_canopy           # 40% of explicit-Euler stability limit

smoothed_signal = sig_fine.copy()
mg.inject_state(smoothed_signal, ['irradiance'])     # abstract API
diffusion_step(mg, 'irradiance', D_canopy, dt, n_steps=20)
sig_smooth = mg.extract_state(['irradiance'])        # abstract API

# ── 4. Plot ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle(
    "MultiGridDataStructure  —  3-level canopy light profile  (Level 3c)\n"
    "FieldDataStructure API delegates to fine grid; restrict/prolongate expose multi-scale structure",
    fontsize=10,
)

# Panel A: signal at each restriction level
#   Restore original signal before plotting
mg.inject_state(sig_fine, ['irradiance'])           # abstract API
plot_multigrid_levels(
    mg, 'irradiance',
    title="Panel A — Restriction cascade\nlow-freq peak preserved; clumping lost",
    ax=axes[0],
)
axes[0].set_xlabel('irradiance (relative)')
axes[0].set_ylabel('canopy height (m)')

# Panel B: original, smoothed, and RP-error
ax = axes[1]
ax.plot(sig_fine,   z_fine, color='steelblue',  linewidth=2.0, label='original (level 0)')
ax.plot(sig_smooth, z_fine, color='darkorange',  linewidth=2.0, linestyle='--',
        label='after diffusion (abstract API)')
ax.plot(error,      z_fine, color='crimson',     linewidth=1.5, linestyle=':',
        label='RP error  (sub-grid detail lost)')
ax.axvline(0, color='grey', linewidth=0.7, linestyle='--')
ax.set_xlabel('irradiance / error')
ax.set_title("Panel B — Restriction–prolongation error\nRP error = detail invisible to coarser levels")
ax.legend(fontsize=8)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'example_multigrid_data_structure.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"Saved → {out}")
