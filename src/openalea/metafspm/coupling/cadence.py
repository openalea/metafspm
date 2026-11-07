"""
A component's cadence: its calls in a scene's steps cover a longer interval::

    GranapAnatomy(data_structure=ds, time_step=time_step, cadence="1 d")    its anatomies once a day
    component.cadence = Cadence("3 h", at="start")                          set after construction too

Cadence(interval, at="end") (or a duration alone, at="end"): the component runs once per *interval* (a whole number of
the scene's steps), its call covering the interval: its clock at the interval's start, its simulation step the
interval (a component running in the scene's step runs one step of the interval; one with its own finer sub-step
sub-steps it), its forcings read over the interval (forcing(over="step")'s, "substep"'s for one step).

  at="end"    it runs at the interval's last step; the inputs it reads through its DataStructure are their means
              over the interval's steps (each entity's over the steps it existed, aligned by id: new and removed
              entities), given under private names during its call (as Link(names=)); its descriptors (kind
              "descriptor") and non-float inputs are their values at the call
  at="start"  it runs at the interval's first step; its inputs are their values then

Between its calls nothing runs: its outputs are held. MPG-style steps read the MTG as it is (their inputs at the
call). The means accumulated so far are kept on the component (Component.keep: saved by checkpoints); the clock is
the Scene's (the component's at each step's start). Component.calls() counts the calls that ran (the Scene's
"calls", next to its timing).
"""
from contextlib import ExitStack, contextmanager

import numpy as np

AT = ("end", "start")
_KEPT = "cadence"


class Cadence:
    """A component's cadence: once per *interval* (a duration: "1 d", "3 h", or s), at the interval's "end"
    (the inputs averaged over it) or "start" (see the module)."""

    def __init__(self, interval, at: str = "end"):
        from openalea.metafspm.coupling.durations import positive_duration
        self.interval = positive_duration(interval, "cadence")
        if at not in AT:
            raise ValueError(f"Cadence(at=...): one of {AT}, got {at!r}")
        self.at = at

    def __eq__(self, other) -> bool:
        return isinstance(other, Cadence) and (self.interval, self.at) == (other.interval, other.at)

    def __hash__(self) -> int:
        return hash((self.interval, self.at))

    def __repr__(self) -> str:
        return f"Cadence({self.interval:g} s, at={self.at!r})"


def cadence_of(component):
    """The component's cadence (its cadence: None, a duration or a Cadence) as a Cadence, or None."""
    cadence = getattr(component, "cadence", None)
    if cadence is None or isinstance(cadence, Cadence):
        return cadence
    return Cadence(cadence)


def _on(time: float, interval: float) -> bool:
    ratio = float(time) / interval
    return abs(ratio - round(ratio)) <= 1e-9 * max(1., abs(ratio))


def _scene_step(component) -> float:
    step = component.simulation_step()
    if step is None:
        raise ValueError(f"{type(component).__name__}(cadence=...): a cadence needs the scene's step (its DataStructure's "
                         "simulation_time_step: run it in a Scene)")
    return float(step)


def run_cadenced(components, cadence: Cadence, run) -> bool:
    """Call *components* (sharing *cadence*) at this scene step: their inputs accumulated (at "end"), then, when the
    interval is due, run(overrides) with their clocks, steps and inputs' means set for the interval
    (overrides: [{input: private name}] per component, the means'); their clocks advanced by the scene's step either
    way. Whether they ran."""
    step = _scene_step(components[0])
    interval = cadence.interval
    from openalea.metafspm.coupling.choregrapher import sub_steps
    sub_steps(interval, step, f"{type(components[0]).__name__}(cadence=)")       # a whole number of the scene's steps
    clocks = [float(component.__dict__.get("_clock", 0.)) for component in components]
    if cadence.at == "end":
        for component in components:
            _accumulate(component, step)
        due, start = _on(clocks[0] + step, interval), clocks[0] + step - interval
    else:
        due, start = _on(clocks[0], interval), clocks[0]
    if due:
        with ExitStack() as stack:
            overrides = [stack.enter_context(_means(component)) if cadence.at == "end" else {}
                         for component in components]
            for component in components:
                stack.enter_context(_covering(component, start, interval, step))
            run(overrides)
        for component in components:
            component.keep(_KEPT, {})
    for component, clock in zip(components, clocks):
        component.__dict__["_clock"] = clock + step
    return due


def _inputs(component) -> list:
    """[(input, stored name)] of the declared inputs to average: float, not descriptors, on its DataStructure."""
    from openalea.metafspm.data_structure.data_api import _root_data_structure
    root = _root_data_structure(component.data_structure)
    private = component.__dict__.get("_private_names") or {}
    found = []
    for name, spec in getattr(component, "_variable_specs", {}).items():
        if spec.variable_type != "input" or spec.kind == "descriptor":
            continue
        stored = private.get(name, name)
        if not root.has(stored) or np.asarray(root.get(stored)).dtype.kind != "f":
            continue
        found.append((name, stored))
    return found


def _aligned(old_ids, old, new_ids, n: int) -> np.ndarray:
    """*old* (rows by *old_ids*) at *new_ids*' rows, 0 for the ids it lacks."""
    out = np.zeros((n,) + old.shape[1:])
    if old_ids is None or new_ids is None:
        return old.copy() if old.shape[0] == n else out
    if old_ids.size and new_ids.size:
        order = np.argsort(old_ids, kind="stable")
        sorted_ids = old_ids[order]
        at = np.minimum(np.searchsorted(sorted_ids, new_ids), sorted_ids.size - 1)
        found = sorted_ids[at] == new_ids
        out[found] = old[order[at[found]]]
    return out


def _accumulate(component, step: float) -> None:
    """Each input's values times *step* added to its total since the interval began, by entity id."""
    from openalea.metafspm.data_structure.data_api import _root_data_structure
    root = _root_data_structure(component.data_structure)
    kept = dict(component.kept(_KEPT) or {})
    for name, stored in _inputs(component):
        location = root.location(stored)
        values = np.asarray(root.get(stored), dtype=float)
        ids = None if location == "scalar" else np.asarray(root.entity_ids(location)).copy()
        n = values.shape[0] if values.ndim else 1
        rows = values.reshape(n, -1)
        total, elapsed = rows * step, np.full(n, step)
        if name in kept:
            old_ids, old_total, old_elapsed = kept[name]
            total = total + _aligned(old_ids, old_total, ids, n)
            elapsed = elapsed + _aligned(old_ids, old_elapsed, ids, n)
        kept[name] = (ids, total, elapsed)
    component.keep(_KEPT, kept)


@contextmanager
def _means(component):
    """The inputs' means over the interval registered under private names during the block (their values now where
    no step was accumulated); yields {input: private name}."""
    from openalea.metafspm.data_structure.data_api import _root_data_structure
    root = _root_data_structure(component.data_structure)
    kept = component.kept(_KEPT) or {}
    names = {}
    try:
        for name, stored in _inputs(component):
            if name not in kept:
                continue
            ids, total, elapsed = kept[name]
            current = np.asarray(root.get(stored), dtype=float)
            location = root.location(stored)
            now = None if location == "scalar" else np.asarray(root.entity_ids(location))
            n = current.shape[0] if current.ndim else 1
            total, elapsed = _aligned(ids, total, now, n), _aligned(ids, elapsed, now, n)
            rows = current.reshape(n, -1)
            mean = np.where(elapsed[:, None] > 0., total / np.where(elapsed > 0., elapsed, 1.)[:, None], rows)
            private = f"_mean_over_cadence.{stored}"           # the same for the components run together
            meta = root._variable_meta().get(root._resolve(stored), {})
            root.register(private, mean.reshape(current.shape), location=location,
                          default=meta.get("default", 0.), shape=tuple(meta.get("shape") or ()))
            root._variable_meta()[private].update({key: value for key, value in meta.items()
                                                   if key not in ("derived", "default", "shape", "dtype")})
            names[name] = private
        yield names
    finally:
        for private in names.values():
            if root.has(private):
                root.unregister(private)


@contextmanager
def _covering(component, start: float, interval: float, step: float):
    """During the block, *component*'s call covers [start, start + interval]: its clock and simulation step, its time
    step the interval when it ran in the scene's step."""
    saved = {key: component.__dict__.get(key) for key in ("_clock", "_choregraphy")}
    time_step = component.time_step
    component.__dict__["_clock"] = start
    component.__dict__["_cadence_step"] = interval
    if time_step is None or abs(float(time_step) - step) <= 1e-9 * step:
        component.time_step = interval
        if saved["_choregraphy"] is not None:
            component.__dict__["_choregraphy"] = (interval, saved["_choregraphy"][1])
    else:
        from openalea.metafspm.coupling.choregrapher import sub_steps
        sub_steps(interval, time_step, type(component).__name__)
    try:
        yield
    finally:
        component.__dict__.pop("_cadence_step", None)
        component.time_step = time_step
        if saved["_choregraphy"] is not None:
            component.__dict__["_choregraphy"] = saved["_choregraphy"]
