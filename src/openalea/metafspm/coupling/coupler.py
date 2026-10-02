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
            total = np.array(self.soil_ds.get(soil_name), dtype=float)
            np.add.at(total.reshape(-1), self.cells, values)
            self.soil_ds.set(soil_name, total)   # through set(): variables derived from it see the write

    def pull(self) -> None:
        self._check_map()
        for plant_name, soil_name in self.to_plant.items():
            self.plant_ds.set(plant_name, self.soil_ds.get(soil_name).reshape(-1)[self.cells])


class Transport:
    """
    Layout of the plant <-> soil shared buffer (one per plant, design note §7): one row per exchanged variable,
    one column per plant node. Two header rows replace the former ``vertex_index >= 1`` convention: the node
    count (``_n_nodes``, column 0) and the node ids (``_node_id``).

    Plant side: write_plant() before each exchange, read_soil() after it.
    Soil side:  plant_view() gives a DataStructure-like view of the buffer on which a Coupler can run.
    """

    COUNT_ROW = "_n_nodes"
    ID_ROW = "_node_id"

    def __init__(self, plant_variables, soil_variables, capacity: int = 20000, to_soil: Mapping = None,
                 to_plant: Mapping = None):
        names = [self.COUNT_ROW, self.ID_ROW] + list(plant_variables)
        names += [name for name in soil_variables if name not in names]
        self.rows = {name: row for row, name in enumerate(names)}
        self.plant_variables = list(plant_variables)
        self.soil_variables = list(soil_variables)
        self.capacity = int(capacity)
        self.to_soil = {name: dict(sources) for name, sources in (to_soil or {}).items()}
        self.to_plant = dict(to_plant or {})

    @property
    def shape(self) -> tuple:
        return (len(self.rows), self.capacity)

    @classmethod
    def from_translator(cls, translator, soil: str, plant_components=None, coordinates=SEGMENT_COORDINATES,
                        capacity: int = 20000) -> "Transport":
        """Rows for the links between *soil* and the plant components (default: every other component)."""
        if plant_components is None:
            plant_components = [name for name in translator.components if name != soil]
        spec = Coupler.from_translator(translator, plant_components, soil, plant_ds=None, soil_ds=None,
                                       locator=VoxelLocator(None, coordinates=coordinates))
        return cls(spec.plant_variables(), spec.soil_variables(), capacity=capacity,
                   to_soil=spec.to_soil, to_plant=spec.to_plant)

    @classmethod
    def from_rows(cls, rows: Mapping, capacity: int, to_soil: Mapping = None, to_plant: Mapping = None) -> "Transport":
        """Transport received from the plant side (rows as sent in the plant message)."""
        names = [name for name, _ in sorted(rows.items(), key=lambda item: item[1])]
        body = [name for name in names if name not in (cls.COUNT_ROW, cls.ID_ROW)]
        soil_variables = [name for name in body if name in set((to_plant or {}).values())]
        plant_variables = [name for name in body if name not in soil_variables]
        transport = cls(plant_variables, soil_variables, capacity=capacity, to_soil=to_soil, to_plant=to_plant)
        if transport.rows != dict(rows):
            raise ValueError("Inconsistent transport rows")
        return transport

    def _count(self, buffer) -> int:
        return int(buffer[self.rows[self.COUNT_ROW], 0])

    def write_plant(self, buffer, plant_ds) -> None:
        n = plant_ds.n_nodes()
        if n > self.capacity:
            raise OverflowError(f"{n} plant nodes exceed the transport capacity of {self.capacity} columns")
        buffer[self.rows[self.COUNT_ROW], 0] = n
        buffer[self.rows[self.ID_ROW], :n] = plant_ds.entity_ids("node")
        for name in self.plant_variables:
            buffer[self.rows[name], :n] = plant_ds.get(name)

    def read_soil(self, buffer, plant_ds, to_plant: Mapping = None) -> None:
        n = self._count(buffer)
        if n != plant_ds.n_nodes():
            raise ValueError(f"The buffer holds {n} nodes, the plant has {plant_ds.n_nodes()}")
        for plant_name, soil_name in (to_plant or self.to_plant).items():
            plant_ds.set(plant_name, buffer[self.rows[soil_name], :n])

    def plant_view(self, buffer) -> "BufferPlantView":
        return BufferPlantView(buffer, self)


class BufferPlantView:
    """Soil-side view of one plant's buffer, with the DataStructure calls a Coupler uses."""

    def __init__(self, buffer, transport: Transport):
        self.buffer = buffer
        self.transport = transport

    def n_nodes(self) -> int:
        return self.transport._count(self.buffer)

    def entity_ids(self, location: str = "node") -> np.ndarray:
        return self.buffer[self.transport.rows[Transport.ID_ROW], :self.n_nodes()].astype(np.int64)

    @property
    def topology_version(self):
        return hash((self.n_nodes(), self.entity_ids().tobytes()))

    def has(self, name: str) -> bool:
        return name in self.transport.rows

    def location(self, name: str) -> str:
        return "node"

    def get(self, name: str) -> np.ndarray:
        return self.buffer[self.transport.rows[name], :self.n_nodes()]

    def set(self, name: str, values) -> None:
        self.buffer[self.transport.rows[name], :self.n_nodes()] = values
