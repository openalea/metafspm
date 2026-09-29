"""
Coupling a plant graph DataStructure with a soil grid DataStructure (design note §7).

A Coupler maps every plant node to the soil cell containing its segment barycenter (VoxelLocator), then
  push(): scatter-adds plant fluxes (extensive, weighted sums of plant variables) into soil cells,
  pull(): gathers the soil state (intensive) of each node's cell into plant variables, in place.
Several plants push into one soil after a single zero_soil_inputs(). The map is rebuilt with update_map(),
at least whenever the plant topology changed.
"""
from typing import Mapping

import numpy as np

SEGMENT_COORDINATES = ("x1", "x2", "y1", "y2", "z1", "z2")


class VoxelLocator:
    """
    Cell of each plant node from its segment end coordinates (plant variables, names configurable).

    flip_z:   plant z is negative below ground while the soil z axis points down (reference soil model).
    periodic: per soil axis (x, y, z), wrap positions into the scene (a periodic stand in x and y by default).
    """

    def __init__(self, grid, coordinates=SEGMENT_COORDINATES, periodic=(True, True, False), flip_z: bool = True):
        self.grid = grid
        self.coordinates = tuple(coordinates)
        self.periodic = periodic
        self.flip_z = flip_z

    def barycenters(self, plant_ds) -> np.ndarray:
        x1, x2, y1, y2, z1, z2 = (np.asarray(plant_ds.get(name), dtype=float) for name in self.coordinates)
        z = 0.5 * (z1 + z2)
        return np.stack([0.5 * (x1 + x2), 0.5 * (y1 + y2), -z if self.flip_z else z], axis=1)

    def cells(self, plant_ds) -> np.ndarray:
        return self.grid.locate(self.barycenters(plant_ds), periodic=self.periodic, clip=True)


class Coupler:
    """
    to_soil:  {soil variable: {plant variable: factor}}, summed into the cell of each plant node.
    to_plant: {plant variable: soil variable}, gathered from the cell of each plant node.
    """

    def __init__(self, plant_ds, soil_ds, locator: VoxelLocator, to_soil: Mapping = None, to_plant: Mapping = None):
        self.plant_ds = plant_ds
        self.soil_ds = soil_ds
        self.locator = locator
        self.to_soil = {name: dict(sources) for name, sources in (to_soil or {}).items()}
        self.to_plant = dict(to_plant or {})
        self.cells = None
        self._map_topology = None

    @classmethod
    def from_translator(cls, translator, plant_components, soil: str, plant_ds, soil_ds, locator: VoxelLocator):
        """Coupler of the links between *soil* and the *plant_components* of a Translator."""
        to_soil, to_plant = {}, {}
        for link in translator.links:
            if link.receiver == soil and link.provider in plant_components:
                if link.formula is not None:
                    raise NotImplementedError(f"formula link {soil}.{link.variable}: not supported across DataStructures")
                to_soil.setdefault(link.variable, {}).update(link.sources)
            elif link.provider == soil and link.receiver in plant_components:
                if link.formula is not None or len(link.sources) != 1 or list(link.sources.values())[0] != 1.:
                    raise NotImplementedError(f"{link.receiver}.{link.variable}: soil states are gathered as is "
                                              "(single source, factor 1)")
                to_plant[link.variable] = next(iter(link.sources))
        return cls(plant_ds, soil_ds, locator, to_soil=to_soil, to_plant=to_plant)

    # ── variables exchanged (transport plan) ──────────────────────────────────

    def plant_variables(self) -> list:
        """Plant variables read by the soil side: segment coordinates, then the sources of to_soil."""
        names = list(self.locator.coordinates)
        for sources in self.to_soil.values():
            names += [name for name in sources if name not in names]
        return names

    def soil_variables(self) -> list:
        """Soil variables sent back to the plant."""
        names = []
        for name in self.to_plant.values():
            if name not in names:
                names.append(name)
        return names

    # ── exchange ──────────────────────────────────────────────────────────────

    def update_map(self) -> None:
        self.cells = self.locator.cells(self.plant_ds)
        self._map_topology = self.plant_ds.topology_version

    def _check_map(self) -> None:
        if self.cells is None or self._map_topology != self.plant_ds.topology_version:
            raise RuntimeError("The plant topology changed since the last update_map(): call update_map() first")

    def zero_soil_inputs(self) -> None:
        """Reset the soil variables receiving plant fluxes, once per step before the plants push."""
        for name in self.to_soil:
            self.soil_ds.set(name, 0.)

    def push(self) -> None:
        self._check_map()
        for soil_name, sources in self.to_soil.items():
            values = sum(float(factor) * np.asarray(self.plant_ds.get(source), dtype=float)
                         for source, factor in sources.items())
            target = self.soil_ds.get(soil_name)
            np.add.at(target.reshape(-1), self.cells, values)

    def pull(self) -> None:
        self._check_map()
        for plant_name, soil_name in self.to_plant.items():
            self.plant_ds.set(plant_name, self.soil_ds.get(soil_name).reshape(-1)[self.cells])
