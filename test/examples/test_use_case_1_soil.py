"""
The soil of examples/use_case_1 (component.RichardsSoil, RichardsSoilModel): Richards' equation in mixed form on
regular and adaptive grids. Its soil curves and their derivatives; the analytic Jacobian against finite differences;
hydrostatic equilibrium (no flux, h - z constant, z the depth); steady drainage to a unit gradient; the water balance
closed to round-off over a run with a root sink and regridding (θ V carried over); and Srivastava & Yeh (1991)'s exact
solution of transient infiltration towards a water table, on a column.
"""
import functools
import os
import sys

import numpy as np
import pytest
from scipy.optimize import brentq

from openalea.metafspm.data_structure.adaptive_grid import AdaptiveGridDataStructure
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.solve.decorator import graph_system

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "use_case_1")
sys.path.insert(0, os.path.abspath(EXAMPLE))

from component import (LOAM, RichardsSoil, RichardsSoilModel, gardner, pressure_head_of,  # noqa: E402
                       scene_translator, set_soil_layers, van_genuchten_mualem)

DAY, HOUR = 86400., 3600.


def _soil(grid, time_step, **settings):
    grid.simulation_time_step = time_step
    return RichardsSoil(data_structure=grid, time_step=time_step, **settings)


def _column(cells, depth=1., width=0.1):
    return ArrayDataStructure(shape=(1, 1, cells), dx=(width, width, depth / cells))


def _values(grid, name):
    return np.asarray(grid.get(name), dtype=float).reshape(-1)


# ── the soil curves ───────────────────────────────────────────────────────────

def test_soil_curves_and_their_derivatives():
    heads = -np.logspace(-3, 1, 40)
    step = 1e-7
    soil = (LOAM["residual_water_content"], LOAM["saturated_water_content"], LOAM["van_genuchten_alpha"],
            LOAM["van_genuchten_n"], LOAM["saturated_conductivity"], 0.5)
    for curves in (lambda h: van_genuchten_mualem(h, *soil),
                   lambda h: gardner(h, soil[0], soil[1], soil[2], soil[4])):
        water_content, capacity, conductivity, slope = curves(heads)
        above, below = curves(heads * (1. + step)), curves(heads * (1. - step))
        np.testing.assert_allclose(capacity, (above[0] - below[0]) / (2. * step * heads), rtol=1e-5,
                                   atol=1e-8 * capacity.max())
        np.testing.assert_allclose(slope, (above[2] - below[2]) / (2. * step * heads), rtol=1e-5,
                                   atol=1e-8 * slope.max())
        assert np.all(np.diff(water_content) < 0.) and np.all(np.diff(conductivity) < 0.)   # drier with suction
        saturated = curves(np.array([0., 0.5]))
        np.testing.assert_allclose(saturated[0], soil[1])
        np.testing.assert_allclose(saturated[2], soil[4])
        np.testing.assert_allclose(saturated[1], 0.)
    for model in ("van_genuchten", "gardner"):
        curves = van_genuchten_mualem(heads, *soil) if model == "van_genuchten" else gardner(
            heads, soil[0], soil[1], soil[2], soil[4])
        wet = heads > -3.                                              # far from θr, where θ - θr loses its digits
        inverse = pressure_head_of(curves[0][wet], soil[0], soil[1], soil[2], soil[3], model)
        np.testing.assert_allclose(inverse, heads[wet], rtol=1e-8)


def test_layers_and_declarations():
    grid = _column(10)
    soil = _soil(grid, HOUR)
    set_soil_layers(grid, [(0.3, {"saturated_conductivity": 1e-5}), (np.inf, {"saturated_conductivity": 1e-6})])
    np.testing.assert_allclose(_values(grid, "saturated_conductivity"), [1e-5] * 3 + [1e-6] * 7)
    table = RichardsSoil.declarations()
    assert {"pressure_head", "plant_uptake", "rainfall", "van_genuchten_n", "face_conductivity"} <= set(table["name"])
    assert soil.parameter_value("water_table_head") == 0.


# ── the analytic Jacobian ─────────────────────────────────────────────────────

@pytest.mark.parametrize("face_conductivity", ["upstream", "arithmetic"])
def test_analytic_jacobian_solves_as_finite_differences(face_conductivity):
    """One solve (no step doubling) of infiltration with a root sink on an adaptive grid (hanging faces): with the
    analytic Jacobian and with a finite-difference one, Newton converges to the same h."""
    options = dict(node_unknowns=["pressure_head", "exchanged_water"], transient=True, integrate="once",
                   tolerance=1e-11, max_iterations=30, linesearch=True, schedule_as="state")

    class AnalyticSoil(RichardsSoil):
        _water = graph_system(solver="newton", **options)(RichardsSoil._water.equations)

    class FiniteDifferenceSoil(RichardsSoil):
        _water = graph_system(solver="newton_fd", **options)(RichardsSoil._water.equations)

    results = []
    for component in (AnalyticSoil, FiniteDifferenceSoil):
        grid = AdaptiveGridDataStructure(shape=(2, 1, 4), dx=(0.1, 0.1, 0.1), max_level=1, periodic=(True, True,
                                                                                                       False))
        grid.refine(np.arange(grid.n_nodes()) == 1)
        grid.simulation_time_step = 600.
        soil = component(data_structure=grid, time_step=600., face_conductivity=face_conductivity,
                         bottom_boundary="water_table")
        soil.set_pressure_head(lambda depth: -0.5 - depth)
        grid.set("rainfall", 2e-6)
        uptake = np.zeros(grid.n_nodes())
        uptake[3] = 1e-9
        grid.set("plant_uptake", uptake)
        soil()
        results.append(_values(grid, "pressure_head"))
    np.testing.assert_allclose(results[0], results[1], atol=1e-6)


# ── equilibria ───────────────────────────────────────────────────────────────

def _adaptive_column():
    grid = AdaptiveGridDataStructure(shape=(2, 2, 10), dx=(0.05, 0.05, 0.1), max_level=1,
                                     periodic=(True, True, False))
    depth = grid.cell_centers()[:, 2]
    grid.refine((depth > 0.3) & (depth < 0.5) & (grid.cell_centers()[:, 0] < 0.05))
    return grid


@pytest.mark.parametrize("grid_kind", ["regular", "adaptive"])
def test_hydrostatic_equilibrium_holds(grid_kind):
    """With a water table at the bottom and no flux at the top, h = h_table - (L - z) is the equilibrium: no flux
    through any face (the hanging faces of an adaptive grid included), h - z unchanged."""
    grid = (ArrayDataStructure(shape=(2, 2, 10), dx=(0.05, 0.05, 0.1), periodic=(True, True, False))
            if grid_kind == "regular" else _adaptive_column())
    soil = _soil(grid, DAY, bottom_boundary="water_table")
    soil.set_pressure_head(lambda depth: -(1. - depth))
    for _ in range(3):
        soil()
    depth = grid.cell_centers()[:, 2]
    np.testing.assert_allclose(_values(grid, "pressure_head") - depth, -1., atol=1e-9)
    assert np.abs(_values(grid, "water_flux")).max() < 1e-16
    assert np.abs(_values(grid, "drainage")).max() < 1e-16


def test_steady_drainage_reaches_a_unit_gradient():
    """A constant rain on a freely draining column: the column wets to K(h) = rain everywhere (a unit gradient of
    the total head, h uniform), and drains what falls."""
    grid = _column(10)
    soil = _soil(grid, 5 * DAY)
    soil.set_pressure_head(-1.)
    rain = 0.3 * LOAM["saturated_conductivity"]
    grid.set("rainfall", rain)
    for _ in range(4):
        soil()
    head = _values(grid, "pressure_head")
    expected = brentq(lambda h: van_genuchten_mualem(h, LOAM["residual_water_content"],
                                                     LOAM["saturated_water_content"], LOAM["van_genuchten_alpha"],
                                                     LOAM["van_genuchten_n"], LOAM["saturated_conductivity"],
                                                     0.5)[2] - rain, -10., -1e-6)
    np.testing.assert_allclose(head, expected, atol=1e-6)
    area = 0.1 * 0.1
    assert _values(grid, "drainage").sum() == pytest.approx(rain * area, rel=1e-6)
    assert _values(grid, "surface_inflow").sum() == pytest.approx(rain * area, rel=1e-12)


# ── the water balance, with a sink and regridding ─────────────────────────────

def test_water_balance_closes_with_uptake_and_regridding():
    """
    RichardsSoilModel on an adaptive grid: rain, evaporation, free drainage and a root sink in one cell, the grid
    refined around the sink and the front and coarsened behind them at each step. The water stored changes by what
    crossed the boundaries and what the roots took, to round-off, over the whole run (θ V carried over at each
    regridding).
    """
    model = RichardsSoilModel([], 0.2, 0.2, HOUR, depth=0.4, voxel=0.1, max_level=1,
                              layers=[(0.2, {**LOAM, "saturated_conductivity": 5e-6}), (np.inf, LOAM)],
                              initial_pressure_head=lambda depth: -1. - depth)
    grid = model.grid
    for step in range(6):
        grid.set("rainfall", 2e-6 if step < 3 else 0.)
        grid.set("potential_evaporation", 0. if step < 3 else 1e-7)
        uptake = np.zeros(grid.n_nodes())
        uptake[grid.locate([[0.05, 0.05, 0.15]])[0]] = 2e-10
        grid.set("plant_uptake", uptake)
        model.run()
        balance = model.water_balance()
        assert balance["change"] == pytest.approx(balance["exchanged"], rel=1e-11)
    levels = np.array(model.history)
    assert (levels[:, 1] > 0).any()                                   # refined at some step
    assert len({grid_cells.sum() for grid_cells in levels}) > 1      # the grid changed
    assert abs(model.water_balance()["change"]) > 1e-5               # a run that moved water


def test_a_rain_forcing_is_followed_within_the_step_and_the_balance_closes():
    """A rain pulse within one day (a forcing, read at each sub-step; the rainfall input left at 0) on a column
    closed at the bottom: the water stored changes by what entered, to round-off, and by the pulse's integral within
    the sub-steps' accuracy; nothing runs off (the rain below Ks)."""
    grid = _column(10)
    soil = _soil(grid, DAY, bottom_boundary="no_flux")
    soil.set_pressure_head(-1.)
    peak, area = 1e-6, 0.1 * 0.1
    soil.set_forcings({"rainfall": ([0., 0.25 * DAY, 0.5 * DAY, 0.75 * DAY], [0., 0., peak, 0.])})
    stored = soil.stored_water()
    soil()
    assert soil._last_integration["steps"] > 4                        # sub-steps following the pulse
    change = soil.stored_water() - stored
    assert change == pytest.approx(soil.exchanged_water_volume(), rel=1e-11)
    assert change == pytest.approx(0.5 * peak * 0.5 * DAY * area, rel=5e-3)
    assert np.all(_values(grid, "rainfall") == 0.) and np.all(_values(grid, "runoff") == 0.)
    soil()                                                              # the next day: dry
    assert soil.stored_water() - stored == pytest.approx(change, rel=1e-11)


def test_carry_over_keeps_the_water_volume():
    """Refining then coarsening a non-uniform h: the water θ V is the same before and after, h recomputed from θ."""
    model = RichardsSoilModel([], 0.2, 0.2, HOUR, depth=0.4, voxel=0.1, max_level=1,
                              initial_pressure_head=lambda depth: -0.2 - 5. * depth ** 2)
    before = model.soil.stored_water()
    model.grid.refine(np.ones(model.grid.n_nodes(), dtype=bool))
    model.soil.carry_over()
    assert model.soil.stored_water() == pytest.approx(before, rel=1e-14)
    model.soil.set_pressure_head(lambda depth: -0.2 - 5. * depth ** 2)
    before = model.soil.stored_water()
    model.grid.coarsen(np.ones(model.grid.n_nodes(), dtype=bool))
    model.soil.carry_over()
    after = model.soil.stored_water()
    model.soil.update_water_content()                                  # θ(h) of the recomputed h
    assert after == pytest.approx(before, rel=1e-14)
    assert model.soil.stored_water() == pytest.approx(before, rel=1e-12)


def test_scene_translator_links_the_plants():
    translator = scene_translator()
    assert {(link.receiver, link.variable) for link in translator.links} == {
        ("MechaHydraulics", "soil_psi"), ("RichardsSoil", "plant_uptake")}


# ── a published benchmark: Srivastava & Yeh (1991) ────────────────────────────

LENGTH, ALPHA, SATURATED, RESIDUAL, CONDUCTIVITY = 1., 10., 0.40, 0.06, 0.01 / HOUR    # m, m-1, -, -, 1 cm h-1
INITIAL_FLUX, FINAL_FLUX = 0.1 * CONDUCTIVITY, 0.9 * CONDUCTIVITY


def srivastava_yeh(height, time, terms=200):
    """
    Srivastava R., Yeh T.-C.J. (1991) Analytical solutions for one-dimensional, transient infiltration toward the
    water table in homogeneous and layered soils. Water Resour. Res. 27: 753-762, doi:10.1029/90WR02772: a Gardner
    soil over a water table (h = 0 at height 0), its surface flux changed at time 0 from INITIAL_FLUX (the steady
    state) to FINAL_FLUX. K = Ks e^(α h) turns Richards' equation into c ∂K/∂t = ∂²K/∂z² + α ∂K/∂z (c = α (θs -
    θr) / Ks, z the height): K is the final steady state plus the series of its decaying eigenmodes,
    e^(-α z / 2) Σ a_n sin(λ_n z) e^(-(λ_n² + α²/4) t / c), tan(λ_n L) = -2 λ_n / α. Returns h at *height*.
    """
    capacity = ALPHA * (SATURATED - RESIDUAL) / CONDUCTIVITY
    final = FINAL_FLUX + (CONDUCTIVITY - FINAL_FLUX) * np.exp(-ALPHA * height)
    series = np.zeros_like(height)
    for eigenvalue, amplitude in _srivastava_yeh_modes(terms):
        series += amplitude * np.sin(eigenvalue * height) * np.exp(-(eigenvalue ** 2 + ALPHA ** 2 / 4.) * time
                                                                   / capacity)
    return np.log((final + np.exp(-ALPHA * height / 2.) * series) / CONDUCTIVITY) / ALPHA


@functools.lru_cache(maxsize=None)
def _srivastava_yeh_modes(terms):
    """The eigenvalues λ_n and the amplitudes a_n of the initial difference (Gauss-Legendre quadrature)."""
    nodes, weights = np.polynomial.legendre.leggauss(400)
    points, weights = (nodes + 1.) * LENGTH / 2., weights * LENGTH / 2.
    initial = np.exp(ALPHA * points / 2.) * (INITIAL_FLUX - FINAL_FLUX) * (1. - np.exp(-ALPHA * points))
    modes = []
    for order in range(1, terms + 1):
        eigenvalue = brentq(lambda value: value * np.cos(value * LENGTH) + ALPHA / 2. * np.sin(value * LENGTH),
                            (order - 0.5) * np.pi / LENGTH + 1e-12, order * np.pi / LENGTH)
        mode = np.sin(eigenvalue * points)
        modes.append((eigenvalue, np.sum(weights * initial * mode) / np.sum(weights * mode ** 2)))
    return tuple(modes)


def _infiltration(cells, face_conductivity, hours):
    grid = _column(cells, LENGTH)
    soil = _soil(grid, HOUR, retention_model="gardner", bottom_boundary="water_table",
                 face_conductivity=face_conductivity)
    for name, value in dict(residual_water_content=RESIDUAL, saturated_water_content=SATURATED,
                            van_genuchten_alpha=ALPHA, saturated_conductivity=CONDUCTIVITY).items():
        grid.set(name, np.full(grid.shape, value))
    height = LENGTH - grid.cell_centers()[:, 2]
    soil.set_pressure_head(srivastava_yeh(height, 0.))
    stored = soil.stored_water()
    grid.set("rainfall", FINAL_FLUX)
    errors = {}
    for hour in range(1, max(hours) + 1):
        soil()
        if hour in hours:
            errors[hour] = np.abs(_values(grid, "pressure_head") - srivastava_yeh(height, hour * HOUR)).max()
    assert soil.stored_water() - stored == pytest.approx(soil.exchanged_water_volume(), rel=1e-10)
    return errors


def test_srivastava_yeh_infiltration_towards_a_water_table():
    """
    The column of Srivastava & Yeh's example (1 m, α = 0.1 cm-1, θs = 0.40, θr = 0.06, Ks = 1 cm h-1, the flux
    raised from 0.1 to 0.9 cm h-1) on 2 cm cells: h within 1 cm of the exact solution with upstream weighting
    (first order in space), within 3 mm with the arithmetic mean, over the 10 h the wetting moves down the column;
    the upstream error decreasing as the cells shrink.
    """
    hours = (2, 5, 10)
    upstream = _infiltration(50, "upstream", hours)
    assert max(upstream.values()) < 0.01
    arithmetic = _infiltration(50, "arithmetic", hours)
    assert max(arithmetic.values()) < 0.003
    coarse = _infiltration(25, "upstream", hours)
    assert all(upstream[hour] < 0.75 * coarse[hour] for hour in hours)
