"""
ArrayDataStructure — illustrative example.

Demonstrates that a generic ``diffusion_step`` function built on the abstract
FieldDataStructure API (coordinates, extract_state, inject_state, laplacian)
drives soil-water redistribution without knowing the concrete class.

Physical scenario
-----------------
1-D soil column, 0–1 m depth, 50 nodes (Δz = 0.02 m).
Initial state: a sharp wetting front — moist layer above 0.3 m, dry below.
The explicit-Euler diffusion integrator redistributes moisture over time.

    ∂θ/∂t = D · L · θ          (method of lines, Neumann BC)

where θ = volumetric water content, D = 1e-4 m²/s (unsaturated diffusivity),
L = second-order FD Laplacian from ds.laplacian().

Two variables are tracked simultaneously through the same abstract interface:
  - 'moisture'     (θ, m³/m³)   — primary variable
  - 'temperature'  (T, °C)      — co-transported tracer

The same diffusion_step and plot_1d_profile functions are called identically
on both variables — the abstract API hides all implementation details.

Output
------
example_array_data_structure.png  (saved next to this file)
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
    FieldDataStructure,
)
from plotting import plot_1d_profile


# ── Generic diffusion integrator — abstract FieldDataStructure API only ───────

def diffusion_step(
    ds: FieldDataStructure,
    var_name: str,
    D: float,
    dt: float,
    n_steps: int = 1,
) -> None:
    """
    Advance ``var_name`` by ``n_steps`` explicit-Euler diffusion steps.

    Uses ONLY the abstract FieldDataStructure API:
        ds.laplacian()        → spatial operator L
        ds.extract_state()    → current field as a flat vector
        ds.inject_state()     → write updated vector back
    """
    L = ds.laplacian()                               # abstract API — cached
    for _ in range(n_steps):
        u   = ds.extract_state([var_name])           # abstract API
        ds.inject_state(u + dt * D * (L @ u), [var_name])   # abstract API


# ── 1. Build the concrete ArrayDataStructure ──────────────────────────────────

n_z   = 50
dz    = 0.02                          # m per node
D_moi = 1e-4                          # moisture diffusivity  (m² s⁻¹)
D_tmp = 5e-5                          # thermal diffusivity   (m² s⁻¹)
dt    = 0.4 * dz**2 / D_moi          # 40% of explicit-Euler stability limit ≈ 1.6 s

ds = ArrayDataStructure(shape=(n_z,), dx=dz, origin=np.array([0.0]))

# Initial moisture: wetting front at 0.3 m  (smooth step)
z = ds.coordinates()[:, 0]
theta0 = 0.35 - 0.25 * (1 + np.tanh((z - 0.30) / 0.04)) / 2
ds.add_field('moisture', theta0)

# Initial temperature: warm surface, cool at depth
temp0  = 22.0 - 8.0 * z / 1.0
ds.add_field('temperature', temp0)

print(f"n_dof  = {ds.n_dof}")                                          # abstract API
print(f"shape  = {ds.shape}")
print(f"L type = {type(ds.laplacian())}")                              # abstract API
print(f"dt     = {dt:.2f} s  (Courant r = {D_moi*dt/dz**2:.3f})")    # < 0.5 → stable

# ── 2. Run diffusion — same function for both variables ───────────────────────

snapshots_moi: dict = {0: ds.extract_state(['moisture']).copy()}     # abstract API
snapshots_tmp: dict = {0: ds.extract_state(['temperature']).copy()}  # abstract API

for checkpoint in [5, 15, 40]:
    steps_since = checkpoint - max(t for t in snapshots_moi if t < checkpoint)
    diffusion_step(ds, 'moisture',    D_moi, dt, n_steps=steps_since)
    diffusion_step(ds, 'temperature', D_tmp, dt, n_steps=steps_since)
    snapshots_moi[checkpoint] = ds.extract_state(['moisture']).copy()
    snapshots_tmp[checkpoint] = ds.extract_state(['temperature']).copy()

print("Diffusion complete.  Final moisture range: "
      f"[{snapshots_moi[40].min():.3f}, {snapshots_moi[40].max():.3f}]")

# ── 3. Plot — plot_1d_profile uses abstract API only ─────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle(
    "ArrayDataStructure  —  1-D soil column diffusion  (Level 3b)\n"
    "diffusion_step and plot_1d_profile use only the abstract FieldDataStructure API",
    fontsize=10,
)

colors    = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
checkpts  = sorted(snapshots_moi)

# Left: moisture profiles
ax = axes[0]
for i, t in enumerate(checkpts):
    # Temporarily inject snapshot into the ds so plot_1d_profile can read it
    ds.inject_state(snapshots_moi[t], ['moisture'])          # abstract API
    plot_1d_profile(ds, 'moisture', xlabel='θ (m³/m³)',
                    label=f't = {t * dt:.0f} s', color=colors[i], ax=ax)
ax.set_title("Moisture θ — redistribution over time")
ax.set_xlim(0.0, 0.42)

# Right: temperature profiles
ax = axes[1]
for i, t in enumerate(checkpts):
    ds.inject_state(snapshots_tmp[t], ['temperature'])       # abstract API
    plot_1d_profile(ds, 'temperature', xlabel='T (°C)',
                    label=f't = {t * dt:.0f} s', color=colors[i], ax=ax)
ax.set_title("Temperature T — co-transported tracer")
ax.set_ylabel('')   # suppress duplicate y-label on shared axis

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'example_array_data_structure.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"Saved → {out}")
