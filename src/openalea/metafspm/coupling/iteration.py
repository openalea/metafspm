"""
The coupling of a scene step: explicit (the default) or iterated to a fixed point.

Explicitly, each model runs once per scene step, after the exchanges into it: a model reads what the models before it
wrote at this step and those after it at the step before (use case 1's soil the plants' uptake of the step before:
decision U-6's lag). A fixed-point coupling iterates chosen schedule steps of several models within the scene step,
until what they write no longer changes::

    Scene(..., coupling=FixedPoint(steps=("RichardsSoilModel.soil", "WheatPlant.hydraulics"), relative_tolerance=1e-8))

After the explicit pass, each iterated step is run again, in the order they ran, from the state it started the scene
step from (its components' kept state and balances, the variables it writes, as they were before its first run), after
the exchanges into its model: a Gauss-Seidel iteration over the exchanges, until every variable the steps write moves
by at most *absolute_tolerance* + *relative_tolerance* × its largest value between two iterations, or *max_iterations* (a warning). A
NaN or an infinity in what they write is no convergence: when the iterations end on one, a FloatingPointError names
the step, the variable and the time. The other steps run once. An iterated step must run every scene step (no cadence) and its DataStructures must keep their
topology from its first run to the iteration (a step that grows or regrids, or a later step of its model doing so,
cannot be iterated: a ValueError names it). Scene.coupling_iterations counts the iterations (run.json).
"""
import copy
import warnings
from dataclasses import dataclass, field

import numpy as np

# The component state a fixed-point iteration restores before running a step again: what keep() kept (its
# checkpointed state, a cadence's means, last_call's values), its conserved quantities' balances and its clock
_RESTORED = ("_kept", "_conservation", "_conservation_start", "_clock")


@dataclass
class FixedPoint:
    """
    A fixed-point coupling of a scene step (see the module): *steps*, the schedule steps iterated, each
    "<model>.<step>" (the model a population's name or an environment model's class name, the step its schedule's:
    Scene.schedule()); *relative_tolerance* and *absolute_tolerance*, the change between two iterations at which the
    iteration stops; *max_iterations*, beyond which it stops with a warning.
    """
    steps: tuple
    relative_tolerance: float = 1e-8
    absolute_tolerance: float = 0.
    max_iterations: int = 20
    _bound: list = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self):
        self.steps = tuple(self.steps)
        if len(self.steps) < 1 or any(not isinstance(name, str) or "." not in name for name in self.steps):
            raise ValueError(f"FixedPoint(steps=...): '<model>.<step>' names, got {self.steps!r}")
        if self.max_iterations < 1 or self.relative_tolerance < 0. or self.absolute_tolerance < 0.:
            raise ValueError("FixedPoint: max_iterations >= 1, relative_tolerance and absolute_tolerance >= 0")

    def bind(self, models: dict) -> None:
        """The iterated steps of *models* ({model name: model}), checked: each known, every scene step (no cadence),
        not structural (it writes no topology)."""
        bound = []
        for name in self.steps:
            model_name, step_name = name.rsplit(".", 1)
            model = models.get(model_name)
            if model is None:
                raise ValueError(f"FixedPoint: no model '{model_name}' in the scene ({sorted(models)})")
            steps = [step for step in model.steps() if step.name == step_name]
            if not steps:
                raise ValueError(f"FixedPoint: '{model_name}' has no step '{step_name}' "
                                 f"({[step.name for step in model.steps()]})")
            for step in steps:
                if step.cadence is not None:
                    raise ValueError(f"FixedPoint: '{name}' runs at a cadence ({step.cadence}): an iterated step "
                                     "runs every scene step")
                if any(variable == "topology" for _, variable in step.writes):
                    raise ValueError(f"FixedPoint: '{name}' changes the topology (a structural step): it cannot be "
                                     "run again from its start")
                bound.append((name, model, step))
        self._bound = bound

    def models(self) -> list:
        """The models holding iterated steps (each once)."""
        out = []
        for _, model, _ in self._bound:
            if all(model is not other for other in out):
                out.append(model)
        return out

    def watches(self, step) -> bool:
        return any(step is bound for _, _, bound in self._bound)


def _roots(step) -> dict:
    from openalea.metafspm.data_structure.data_api import _root_data_structure
    roots = {}
    for component in step.components:
        root = _root_data_structure(component.data_structure)
        roots[id(root)] = root
    return roots


def _written(step) -> list:
    """(root DataStructure, name) of the stored float variables *step* writes."""
    out = []
    for root, name in sorted(step.writes, key=lambda access: (id(access[0]), str(access[1]))):
        if name in ("topology", "geometry") or not hasattr(root, "has") or not root.has(name):
            continue
        values = np.asarray(root.get(name))
        if values.dtype.kind == "f":
            out.append((root, name))
    return out


def _snapshot(step) -> dict:
    """What a fixed-point iteration restores before running *step* again: its components' state (_RESTORED), the
    variables it writes, the topology versions of their DataStructures."""
    return {"components": [{key: copy.deepcopy(component.__dict__[key]) for key in _RESTORED
                            if key in component.__dict__} for component in step.components],
            "variables": [(root, name, np.array(root.get(name), copy=True)) for root, name in _written(step)],
            "topology": {key: root.topology_version for key, root in _roots(step).items()}}


def _restore(step, kept: dict, name: str) -> None:
    """Restore *step*'s state from *kept* (_snapshot()), a ValueError when its DataStructures' topology changed."""
    for key, root in _roots(step).items():
        if root.topology_version != kept["topology"].get(key, root.topology_version):
            raise ValueError(f"FixedPoint: the topology of '{name}''s DataStructure changed after it ran (growth, a "
                             "regridding: a later step of its model, or of a model after it): it cannot be run again "
                             "from its start; iterate steps whose DataStructures keep their topology")
    for component, state in zip(step.components, kept["components"]):
        for key in _RESTORED:
            component.__dict__.pop(key, None)
        component.__dict__.update(copy.deepcopy(state))
    for root, variable, values in kept["variables"]:
        root.set(variable, values)


def values(bound) -> list:
    """The variables the iterated steps write, copies, in order."""
    return [np.array(root.get(name), dtype=float, copy=True) for _, _, step in bound for root, name in _written(step)]


def names(bound) -> list:
    """("<model>.<step>", variable) of the variables the iterated steps write, in values()' order."""
    return [(step_name, name) for step_name, _, step in bound for _, name in _written(step)]


def converged(before: list, after: list, relative_tolerance: float, absolute_tolerance: float) -> bool:
    """Whether every variable moved by at most absolute_tolerance + relative_tolerance × its largest value; a NaN or an infinity in
    either iterate is no convergence."""
    for old, new in zip(before, after):
        if old.shape != new.shape:
            return False
        if not old.size:
            continue
        if not (np.isfinite(old).all() and np.isfinite(new).all()):
            return False
        if np.max(np.abs(new - old)) > absolute_tolerance + relative_tolerance * np.max(np.abs(new)):
            return False
    return True


def check_finite(bound, after: list, time: float) -> None:
    """A FloatingPointError naming the first iterated step and variable holding a NaN or an infinity in *after*."""
    for (step_name, name), new in zip(names(bound), after):
        bad = ~np.isfinite(new)
        if bad.any():
            raise FloatingPointError(f"FixedPoint: '{step_name}' wrote a non-finite '{name}' ({int(bad.sum())} of "
                                     f"{new.size} values NaN or infinite) at time {time} s: the iteration cannot "
                                     "converge")


def warn_not_converged(coupling: FixedPoint, time: float) -> None:
    warnings.warn(f"FixedPoint: not converged in {coupling.max_iterations} iterations at time {time} s "
                  f"(steps {list(coupling.steps)}, relative tolerance {coupling.relative_tolerance})")
