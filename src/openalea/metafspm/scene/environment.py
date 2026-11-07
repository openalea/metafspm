"""
The base of a scene's environment models (light, soil, atmosphere)::

    class Soil(EnvironmentModel):
        compartment = "Soil"                                        # the folder of its outputs
        schedule = ("transport", "adapt")                           # its steps: a component, then a method
        def __init__(self, plant_data_structures, xrange, yrange, time_step, depth=0.4, voxel=0.025):
            super().__init__(plant_data_structures, xrange, yrange, time_step)
            self.grid = self.build_grid(voxel, depth, adaptive=True)  # a grid over the scene's pattern, z down
            self.transport = SoilWaterTransport(data_structure=self.grid, time_step=time_step)
            self.components = [self.transport]
        def adapt(self):
            ...                                                     # the grid refined where the fluxes are

    class Light(EnvironmentModel):
        compartment = "Atmosphere"
        schedule = (Step("light", stages="rate", cadence=Cadence("4 h", at="start")),    # lit every 4 hours
                    Step("light", stages="state", name="absorbed_light"))               # its PARa every step
        def __init__(self, plant_data_structures, xrange, yrange, time_step):
            super().__init__(plant_data_structures, xrange, yrange, time_step)
            self.light = CaribuLight(data_structure=self.plant_data_structure(), time_step=time_step)
            self.components = [self.light]

The Scene builds it with its populations' DataStructures, the scene's ranges and time step (and the scenario's keys
its __init__ names), then calls run() at every step: its schedule (coupling.schedule: its components, by their
attribute names, and its methods; default: its components in order), each step at its cadence (a Step's cadence=,
the scenario's "cadence" {step name: duration}). A plain class with the same contract (plant_data_structures,
xrange, yrange, time_step; components; run) is still accepted by the Scene.
"""
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.cross import UnionDataStructure


def _scale_name(ds, scale) -> str:
    return next(name for name, value in vars(type(ds.structure.scales)).items()
                if isinstance(value, int) and not name.startswith("_") and value == scale)


class EnvironmentModel(CompositeModel):
    """An environment model of a Scene: its DataStructures over the scene's pattern, its components, its schedule."""

    # The folder of its outputs (SceneRecorder): "Soil" or "Atmosphere", required
    compartment = None

    def __init__(self, plant_data_structures, xrange, yrange, time_step):
        self.plant_data_structures = list(plant_data_structures)
        self.xrange, self.yrange, self.time_step = xrange, yrange, time_step
        # the scene's pattern (x1, y1, x2, y2), repeated in x and y by periodic models
        self.pattern = (0., 0., float(xrange), float(yrange)) if xrange is not None else None
        self.components = []
        self._plants = None

    # ── its DataStructures ───────────────────────────────────────────────────

    def plant_data_structure(self):
        """The plants' elements, one DataStructure for the model's components: the union of the populations' (of one
        population too), an anatomy-mode population's elements its sections' owners (its view at that scale); their
        variables reach the plant models through the scene translator's links (identity ones by default, Scene). Built
        once."""
        if self._plants is None:
            parts = [ds.at(_scale_name(ds, ds.from_scale)) if getattr(ds, "_anatomy", False) else ds
                     for ds in self.plant_data_structures]
            self._plants = UnionDataStructure(parts)
        return self._plants

    def build_grid(self, voxel, depth, adaptive=False, max_level=2, periodic=(True, True, False)):
        """A grid over the scene's pattern down to *depth* (z pointing down from the surface), of cells of about
        *voxel* (whole numbers of cells per axis), periodic in x and y by default; an AdaptiveGridDataStructure
        refinable *max_level* times with *adaptive*."""
        from openalea.metafspm.data_structure.adaptive_grid import AdaptiveGridDataStructure
        from openalea.metafspm.data_structure.data_api import ArrayDataStructure
        x_range, y_range = float(self.xrange), float(self.yrange)
        shape = (max(1, round(x_range / voxel)), max(1, round(y_range / voxel)), max(1, round(depth / voxel)))
        dx = (x_range / shape[0], y_range / shape[1], depth / shape[2])
        if adaptive:
            return AdaptiveGridDataStructure(shape=shape, dx=dx, periodic=periodic, max_level=max_level)
        return ArrayDataStructure(shape=shape, dx=dx, periodic=periodic)

    # ── its time ─────────────────────────────────────────────────────────────

    def time(self) -> float:
        """The scene's time at this step's start (the Scene sets it on the components before running the model)."""
        return float(self.components[0].__dict__.get("_clock", 0.)) if self.components else 0.

    # ── checkpoints ──────────────────────────────────────────────────────────

    def checkpoint_state(self) -> dict:
        """The model's own state for the Scene's checkpoints (its components' are theirs): none by default."""
        return {}

    def restore_state(self, state: dict) -> None:
        """The model's own state back from a checkpoint (checkpoint_state's)."""
