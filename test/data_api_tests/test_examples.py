"""
Smoke tests of the illustrative examples: a 1-D soil column diffusion driven through the abstract
FieldDataStructure API, and the plots of a plant graph and its incidence. Images are written to a temporary folder.
"""
import os
import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openalea.metafspm.data_structure.data_api import ArrayDataStructure, FieldDataStructure, MPGDataStructure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))
from ds_plotting import plot_1d_profile, plot_graph_property, plot_incidence_structure

from simple_seedling import generate_simple_mpg_seedling


def diffusion_step(ds: FieldDataStructure, name: str, diffusivity: float, dt: float, n_steps: int = 1) -> None:
    """Explicit-Euler diffusion of *name*, through the abstract API only (laplacian, extract_state, inject_state)."""
    L = ds.laplacian()
    for _ in range(n_steps):
        u = ds.extract_state([name])
        ds.inject_state(u + dt * diffusivity * (L @ u), [name])


def test_soil_column_diffusion_through_the_abstract_field_api(tmp_path):
    """A wetting front and a temperature gradient relax, conserving their totals (no-flux ends)."""
    n_z, dz, d_moisture, d_temperature = 50, 0.02, 1e-4, 5e-5
    dt = 0.4 * dz ** 2 / d_moisture                         # within the explicit stability limit
    ds = ArrayDataStructure(shape=(n_z,), dx=dz, origin=np.array([0.0]))
    z = ds.coordinates()[:, 0]
    ds.register("moisture", 0.35 - 0.25 * (1 + np.tanh((z - 0.30) / 0.04)) / 2)
    ds.register("temperature", 22.0 - 8.0 * z)
    before = {name: ds.extract_state([name]).copy() for name in ("moisture", "temperature")}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for name, diffusivity, ax in (("moisture", d_moisture, axes[0]), ("temperature", d_temperature, axes[1])):
        plot_1d_profile(ds, name, label="t = 0", ax=ax)
        diffusion_step(ds, name, diffusivity, dt, n_steps=40)
        plot_1d_profile(ds, name, label=f"t = {40 * dt:.0f} s", ax=ax)
    fig.savefig(tmp_path / "soil_column.png")
    plt.close(fig)

    for name, initial in before.items():
        after = ds.extract_state([name])
        np.testing.assert_allclose(after.sum(), initial.sum(), rtol=1e-12)           # no-flux ends conserve
        assert np.abs(np.diff(after)).max() < np.abs(np.diff(initial)).max()        # the profile smooths
    assert (tmp_path / "soil_column.png").stat().st_size > 0


def test_plant_graph_plots(tmp_path):
    """A seedling's graph coloured by a node variable, and its incidence matrix."""
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    ds.register("water_potential", -np.linspace(0.1, 1.0, ds.n_nodes()), location="node")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_graph_property(ds, "water_potential", ax=axes[0])
    plot_incidence_structure(ds, ax=axes[1])
    fig.savefig(tmp_path / "plant_graph.png")
    plt.close(fig)
    assert (tmp_path / "plant_graph.png").stat().st_size > 0
