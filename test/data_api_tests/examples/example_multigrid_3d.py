"""
MultiGridDataStructure (3D) — soil volume illustrative example.

Two demonstrations on a 3-D soil voxel grid (12 × 12 × 8, 1.2 m × 1.2 m × 0.4 m):

Part 1 — Fine vs coarse resolution
────────────────────────────────────
An ArrayDataStructure at full resolution is compared with a MultiGridDataStructure
whose restriction / prolongation operators are built as proper 3-D Kronecker products:

    R_3d = R_x ⊗ R_y ⊗ R_z       (preserves C-order ravelling of (nx, ny, nz) arrays)
    P_3d = P_x ⊗ P_y ⊗ P_z

The three subplots (left → right) for a horizontal surface slice show:
  (1) Original fine-scale temperature — two Gaussian hot/cool spots + fine noise
  (2) Coarse approximation  P( R( fine ) )  — noise removed, spots blurred
  (3) Sub-grid detail = fine − coarse_approx — what coarse level cannot represent

Part 2 — Variable communication between grid levels
─────────────────────────────────────────────────────
A coarse-scale atmospheric forcing ('radiation', W m⁻²) is computed on the coarse
grid (6 × 6 × 4).  It is prolongated to the fine grid through the abstract API:

    mg.prolongate(radiation_coarse, to_level=0)     # abstract API

The fine soil model then absorbs the prolongated forcing in one explicit step,
updating the fine-scale temperature field.

This represents the domain-decomposition coupling pattern used in FSPM:
  - coarse atmospheric model supplies boundary data at coarse resolution
  - fine plant / soil model refines it through P and solves locally
  - fine model summary is restricted back to the coarse model via R (not shown here)

Abstract API usage summary
──────────────────────────
All consumer functions (diffusion_step, plot_*, the forcing loop) see only:
    FieldDataStructure: shape, n_dof, coordinates, laplacian, extract_state, inject_state
    MultiGridDataStructure: the above + restrict, prolongate, n_levels

Output
──────
example_multigrid_3d.png  (saved next to this file)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from openalea.metafspm.data_structure.data_api import (
    ArrayDataStructure,
    MultiGridDataStructure,
    GridLevel,
    FieldDataStructure,
)
from plotting import plot_1d_profile


# ═══════════════════════════════════════════════════════════════════════════════
# Generic helpers — abstract FieldDataStructure API only
# ═══════════════════════════════════════════════════════════════════════════════

def diffusion_step(
    ds: FieldDataStructure, var_name: str, D: float, dt: float, n_steps: int = 1,
) -> None:
    """Abstract API: laplacian, extract_state, inject_state."""
    L = ds.laplacian()
    for _ in range(n_steps):
        u = ds.extract_state([var_name])
        ds.inject_state(u + dt * D * (L @ u), [var_name])


def _slice_xy(ds: FieldDataStructure, var_name: str, iz: int) -> np.ndarray:
    """Extract a horizontal (x-y) slice at z-index iz via abstract API."""
    nx, ny, nz = ds.shape
    return ds.extract_state([var_name]).reshape(nx, ny, nz)[:, :, iz]


# ═══════════════════════════════════════════════════════════════════════════════
# Grid construction
# ═══════════════════════════════════════════════════════════════════════════════

nx, ny, nz   = 12, 12, 8      # fine  1.2 m × 1.2 m × 0.4 m
nxc, nyc, nzc = 6,  6,  4    # coarse (factor 2 in every dimension)
dx, dz        = 0.10, 0.05   # m (fine spacing)

fine   = ArrayDataStructure(shape=(nx, ny, nz),    dx=np.array([dx, dx, dz]))
coarse = ArrayDataStructure(shape=(nxc, nyc, nzc), dx=np.array([dx*2, dx*2, dz*2]))

# ── Proper 3-D Kronecker restriction / prolongation ───────────────────────────

R_x  = MultiGridDataStructure._build_restriction (nx,  nxc, 2)   # (6,  12)
R_y  = MultiGridDataStructure._build_restriction (ny,  nyc, 2)   # (6,  12)
R_z  = MultiGridDataStructure._build_restriction (nz,  nzc, 2)   # (4,   8)
R_3d = np.kron(np.kron(R_x, R_y), R_z)                           # (144, 1152)
# Normalise rows so boundary coarse nodes are proper weighted averages.
# Without this, Kronecker boundary nodes (row sum = 0.75^3 at grid corners)
# systematically under-represent the restricted field.
_row_sums = R_3d.sum(axis=1, keepdims=True)
R_3d = R_3d / np.where(_row_sums > 1e-12, _row_sums, 1.0)

P_x  = MultiGridDataStructure._build_prolongation(nx,  nxc)      # (12,  6)
P_y  = MultiGridDataStructure._build_prolongation(ny,  nyc)      # (12,  6)
P_z  = MultiGridDataStructure._build_prolongation(nz,  nzc)      # ( 8,  4)
P_3d = np.kron(np.kron(P_x, P_y), P_z)                           # (1152, 144)

print(f"R_3d  {R_3d.shape}   P_3d  {P_3d.shape}")
print(f"Fine DOF = {fine.n_dof}   Coarse DOF = {coarse.n_dof}")

mg = MultiGridDataStructure([
    GridLevel(grid=fine,   level=0, restriction=None,  prolongation=None),
    GridLevel(grid=coarse, level=1, restriction=R_3d,  prolongation=P_3d),
])
print(f"MultiGrid  n_levels={mg.n_levels}  shape={mg.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — Initial 3-D temperature field
# ═══════════════════════════════════════════════════════════════════════════════

coords = fine.coordinates()                  # abstract API  (1152, 3)
X = coords[:, 0].reshape(nx, ny, nz)
Y = coords[:, 1].reshape(nx, ny, nz)
Z = coords[:, 2].reshape(nx, ny, nz)

# Low-frequency signal: two Gaussian blobs at surface + linear depth gradient
np.random.seed(7)
T_low = (20.0
         + 7.0 * np.exp(-((X - 0.3)**2 + (Y - 0.3)**2) / 0.04) * np.exp(-Z / 0.15)
         - 5.0 * np.exp(-((X - 0.9)**2 + (Y - 0.9)**2) / 0.04) * np.exp(-Z / 0.20)
         - 6.0 * Z)                           # cooling with depth

# High-frequency noise (sub-grid detail: only visible at fine resolution)
T_noise = 0.6 * np.random.randn(nx, ny, nz)

T_init = T_low + T_noise
fine.add_field('temperature', T_init)        # concrete setup — from here: abstract API only

# ── Restrict to coarse, then prolongate back → coarse approximation ───────────

T_flat   = fine.extract_state(['temperature'])          # abstract API
T_coarse = mg.restrict(T_flat, from_level=0)            # abstract API  R_3d @ T_flat
T_back   = mg.prolongate(T_coarse, to_level=0)          # abstract API  P_3d @ T_coarse
detail   = T_flat - T_back                              # sub-grid information

print(f"Fine     ψ range [{T_flat.min():.2f}, {T_flat.max():.2f}] °C")
print(f"Coarse   ψ range [{T_back.min():.2f}, {T_back.max():.2f}] °C")
print(f"Detail   max|e| = {np.abs(detail).max():.3f}  RMS = {np.sqrt((detail**2).mean()):.3f} °C")

# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Cross-level forcing: coarse → fine variable communication
# ═══════════════════════════════════════════════════════════════════════════════

# Coarse atmospheric model computes solar radiation input (W m⁻²)
Xc = coarse.coordinates()[:, 0].reshape(nxc, nyc, nzc)
Yc = coarse.coordinates()[:, 1].reshape(nxc, nyc, nzc)
R_rad = (120.0 * np.exp(-((Xc - 0.6)**2 + (Yc - 0.9)**2) / 0.10)
         + 60.0 * np.exp(-((Xc - 0.3)**2 + (Yc - 0.4)**2) / 0.05))
coarse.add_field('radiation', R_rad)

# Communicate to fine solver via abstract API
rad_coarse = coarse.extract_state(['radiation'])           # abstract API on coarse DS
rad_fine   = mg.prolongate(rad_coarse, to_level=0)        # abstract API on multigrid

# Fine model absorbs prolongated forcing (one explicit source step)
alpha     = 5e-4    # K per (W m⁻²) per step — simplified absorption
T_driven  = T_flat + alpha * rad_fine                     # source-driven update
fine.inject_state(T_driven, ['temperature'])              # abstract API

# Optional: one diffusion step to spread the surface heating
dz_min = float(np.min(fine._dx))
n_dims = len(fine.shape)
D_soil = 5e-4                             # m² s⁻¹
dt     = 0.3 * dz_min**2 / (n_dims * D_soil)
diffusion_step(mg, 'temperature', D_soil, dt, n_steps=5)  # abstract API via mg
T_final = mg.extract_state(['temperature'])                # abstract API

print(f"After coupling + diffusion: T range [{T_final.min():.2f}, {T_final.max():.2f}] °C")

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting — 2 rows × 3 columns
# ═══════════════════════════════════════════════════════════════════════════════

# Horizontal extents for imshow
ext_fine   = [0, nx*dx,  0, ny*dx]
ext_coarse = [0, nxc*dx*2, 0, nyc*dx*2]

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "MultiGridDataStructure (3D)  —  soil volume 1.2 m × 1.2 m × 0.4 m  (Level 3c)\n"
    "Proper Kronecker R_3d = R_x ⊗ R_y ⊗ R_z;  mg.restrict / mg.prolongate via abstract API",
    fontsize=11,
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Common temperature scale for row 1 (fine and coarse approx share scale)
vmin_T = T_flat.min();  vmax_T = T_flat.max()
vabs_d = np.abs(detail).max()

def _imshow(ax, data, ext, cmap, vmin, vmax, title, cbar_label):
    im = ax.imshow(data.T, origin='lower', extent=ext,
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal',
                   interpolation='bilinear')
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.75, pad=0.02)
    ax.set_xlabel('x (m)');  ax.set_ylabel('y (m)')
    ax.set_title(title)
    return im

# ── Row 1: fine vs coarse comparison ─────────────────────────────────────────

iz_show = 0   # surface slice

ax00 = fig.add_subplot(gs[0, 0])
_imshow(ax00, T_flat.reshape(nx, ny, nz)[:, :, iz_show],
        ext_fine, 'RdYlBu_r', vmin_T, vmax_T,
        f"Fine grid  ({nx}×{ny}×{nz})  z = {iz_show*dz:.2f} m",
        "T (°C)")

ax01 = fig.add_subplot(gs[0, 1])
_imshow(ax01, T_back.reshape(nx, ny, nz)[:, :, iz_show],
        ext_fine, 'RdYlBu_r', vmin_T, vmax_T,
        f"Coarse approx  P(R(·))  z = {iz_show*dz:.2f} m",
        "T (°C)")
ax01.set_title(ax01.get_title() + "\n(high-freq noise removed)", fontsize=8)

ax02 = fig.add_subplot(gs[0, 2])
_imshow(ax02, detail.reshape(nx, ny, nz)[:, :, iz_show],
        ext_fine, 'RdBu', -vabs_d, vabs_d,
        "Sub-grid detail = fine − P(R(fine))\n(invisible to coarse solver)",
        "ΔT (°C)")

# ── Row 2: cross-level forcing / variable communication ──────────────────────

vmin_r = min(rad_coarse.min(), rad_fine.min())
vmax_r = max(rad_coarse.max(), rad_fine.max())

ax10 = fig.add_subplot(gs[1, 0])
_imshow(ax10, rad_coarse.reshape(nxc, nyc, nzc)[:, :, 0],
        ext_coarse, 'YlOrRd', vmin_r, vmax_r,
        f"Coarse forcing  ({nxc}×{nyc}×{nzc})  z = 0\n(atmosphere model output)",
        "radiation (W m⁻²)")

ax11 = fig.add_subplot(gs[1, 1])
_imshow(ax11, rad_fine.reshape(nx, ny, nz)[:, :, iz_show],
        ext_fine, 'YlOrRd', vmin_r, vmax_r,
        f"Prolongated to fine  ({nx}×{ny}×{nz})  z = 0\n"
        "mg.prolongate(rad_coarse, to_level=0)",
        "radiation (W m⁻²)")

ax12 = fig.add_subplot(gs[1, 2])
_imshow(ax12, T_final.reshape(nx, ny, nz)[:, :, iz_show],
        ext_fine, 'RdYlBu_r', None, None,
        f"Fine T after source + diffusion  z = 0\n"
        "coarse-to-fine coupling via abstract API",
        "T (°C)")

plt.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(os.path.dirname(__file__), 'example_multigrid_3d.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"Saved → {out}")
