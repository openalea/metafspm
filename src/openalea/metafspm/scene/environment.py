"""
The base of a scene's environment models (light, soil, atmosphere):

    class Soil(EnvironmentModel):
        def __init__(self, populations, scene_xrange, scene_yrange, time_step, depth=0.4, voxel=0.025):
            super().__init__(populations, scene_xrange, scene_yrange, time_step)
            self.grid = self.build_grid(voxel, depth)               # a grid over the scene's pattern, z down
            self.transport = SoilWaterTransport(data_structure=self.grid, time_step=time_step)
            self.components = [self.transport]

    class Light(EnvironmentModel):
        interval = 4 * 3600.                                        # a full run every 4 hours
        def __init__(self, populations, scene_xrange, scene_yrange, time_step, **parameters):
            super().__init__(populations, scene_xrange, scene_yrange, time_step)
            self.light = CaribuLight(data_structure=self.plants(), time_step=time_step, **parameters)
            self.components = [self.light]
        def between(self):
            self.light.run_stage("state")                           # the other steps: only the row "state"

The Scene builds it with its populations' DataStructures, the scene's ranges and time step (and the scenario's keys
its __init__ names), then calls run() at every step. run() runs the components in order when the model is due
(every step, or every *interval*, or an overridden due()), then after_run() (e.g. regridding); otherwise between().
The time of the last full run is kept by checkpoints. A plain class with the same contract (populations,
scene_xrange, scene_yrange, time_step; components; run) is still accepted by the Scene.
"""
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.cross import UnionDataStructure


class EnvironmentModel(CompositeModel):
    """An environment model of a Scene: its DataStructures over the scene's pattern, its components, its cadence."""

    # Seconds between two full runs (None: every step); between them, between() runs instead
    interval = None

    def __init__(self, populations, scene_xrange, scene_yrange, time_step):
        self.populations = list(populations)
        self.scene_xrange, self.scene_yrange, self.time_step = scene_xrange, scene_yrange, time_step
        # the scene's pattern (x1, y1, x2, y2), repeated in x and y by periodic models
        self.pattern = (0., 0., float(scene_xrange), float(scene_yrange)) if scene_xrange is not None else None
        self.components = []
        self.last_run = None
        self._plants = None

    # ── its DataStructures ───────────────────────────────────────────────────

    def plants(self):
        """The plants' DataStructure: the population's, or the union of several (a light model seeing every
        plant); built once."""
        if self._plants is None:
            self._plants = (self.populations[0] if len(self.populations) == 1
                            else UnionDataStructure(self.populations))
        return self._plants

    def build_grid(self, voxel, depth, adaptive=False, max_level=2, periodic=(True, True, False)):
        """A grid over the scene's pattern down to *depth* (z pointing down from the surface), of cells of about
        *voxel* (whole numbers of cells per axis), periodic in x and y by default; an AdaptiveGridDataStructure
        refinable *max_level* times with *adaptive*."""
        from openalea.metafspm.data_structure.adaptive_grid import AdaptiveGridDataStructure
        from openalea.metafspm.data_structure.data_api import ArrayDataStructure
        x_range, y_range = float(self.scene_xrange), float(self.scene_yrange)
        shape = (max(1, round(x_range / voxel)), max(1, round(y_range / voxel)), max(1, round(depth / voxel)))
        dx = (x_range / shape[0], y_range / shape[1], depth / shape[2])
        if adaptive:
            return AdaptiveGridDataStructure(shape=shape, dx=dx, periodic=periodic, max_level=max_level)
        return ArrayDataStructure(shape=shape, dx=dx, periodic=periodic)

    # ── its cadence ──────────────────────────────────────────────────────────

    def time(self) -> float:
        """The scene's time at this step's start (the Scene sets it on the components before running the model)."""
        return float(self.components[0].__dict__.get("_clock", 0.)) if self.components else 0.

    def due(self, time) -> bool:
        """Whether the model runs fully at *time*: every step, or every *interval*."""
        return self.interval is None or self.last_run is None or time - self.last_run >= self.interval - 1e-9

    def run(self) -> None:
        time = self.time()                  # before the components' calls, which advance their clocks
        if self.due(time):
            self.full_run()
            self.last_run = time
            self.after_run()
        else:
            self.between()

    def full_run(self) -> None:
        """The components, in order."""
        for component in self.components:
            component()

    def between(self) -> None:
        """What runs at the steps when the model is not due (nothing by default)."""

    def after_run(self) -> None:
        """What follows a full run (e.g. an adaptive grid refined where the fluxes are)."""

    # ── checkpoints ──────────────────────────────────────────────────────────

    def checkpoint_state(self) -> dict:
        return {"last_run": self.last_run}

    def restore_state(self, state: dict) -> None:
        self.last_run = state.get("last_run")
