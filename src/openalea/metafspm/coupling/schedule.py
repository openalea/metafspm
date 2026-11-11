"""
A model's schedule: the steps of its run, declared with its parts (coupling.composite_wrapper.CompositeModel)::

    class WheatPlant(CompositeModel):
        parts = {"crown": WheatCrown, "shoot": ScenarioShoot, "rule": NodalRootEmergence, "roots": RootArchitecture,
                 "anatomy": GranapAnatomy, ...}
        schedule = ("shoot", "rule", "roots", Step("anatomy", cadence="1 d"), ...)

    class Light(EnvironmentModel):
        schedule = (Step("light", stages="rate", cadence=Cadence("4 h", at="start")),     # lit every 4 hours
                    Step("light", stages="state", name="absorbed_light"))                # its PARa every step

A schedule entry is:
  "name"                  a part (a component: its call; a nested model: its own schedule) or a method of the model
                          (a step over all the model's parts, e.g. ScenarioShoot's grow loop, a soil's adapt);
  Step(name, stages=, cadence=, name=)
                          the same, with *stages* (a component's Choregrapher stages, e.g. "rate" or ("rate",
                          "state"): only those of its steps, run_stage's rows) and a *cadence* (a duration, "1 d", or
                          a coupling.cadence.Cadence: its calls cover the interval), named *name* (default: the part's
                          name; a part scheduled twice names its other steps);
  Together(name, ..., cadence=, name=)
                          components run as one (CompositeModel.run_together: their step groups by schedule position).

Without a schedule, the parts run in their order (an environment model: its components), those without steps (an
initiator alone) left out. A model's scenario key
"cadence" ({step name: duration or Cadence}; None: every step) sets its steps' cadences, a duration keeping the
declared Cadence's "at" (the benchmark's --cadence). A cadence of the scene's step or shorter is every step.

Each step's reads and writes are explicit (ScheduledStep.reads, .writes; CompositeModel.steps(), Scene.schedule()):
pairs (DataStructure, variable) of the variables stored, the inputs resolved through the DataStructure's aliases,
derivations and the receiver's private names, the states written; "topology" written by a structural component with
steps and read by every component of its DataStructure, "geometry" read by a geometry input (geometry_input) and
written by the structural components; a method step's are all its model's parts'. Two steps conflict when one writes
what the other reads or writes (conflicts()): the steps without conflict may run concurrently
(devplan_scene_parallel P4).
"""
from time import perf_counter

STRUCTURE, GEOMETRY = "topology", "geometry"


class Step:
    """A schedule entry: the part or method *target*, its *stages* (None: all), its *cadence* (None: every step), its
    *name* (default: *target*)."""

    def __init__(self, target: str, stages=None, cadence=None, name: str = None):
        if not isinstance(target, str):
            raise TypeError(f"Step: a part's or a method's name, got {target!r}")
        self.targets = (target,)
        self.stages = (stages,) if isinstance(stages, str) else (tuple(stages) if stages is not None else None)
        self.cadence = cadence
        self.name = name or target

    def __repr__(self) -> str:
        return f"Step({self.targets[0]!r}, stages={self.stages}, cadence={self.cadence!r}, name={self.name!r})"


class Together(Step):
    """A schedule entry: the components *targets* run as one (CompositeModel.run_together)."""

    def __init__(self, *targets, cadence=None, name: str = None):
        if len(targets) < 2 or not all(isinstance(target, str) for target in targets):
            raise TypeError(f"Together: two parts' names or more, got {targets!r}")
        self.targets, self.stages, self.cadence = tuple(targets), None, cadence
        self.name = name or "+".join(targets)

    def __repr__(self) -> str:
        return f"Together({', '.join(map(repr, self.targets))}, cadence={self.cadence!r}, name={self.name!r})"


class ScheduledStep:
    """
    A step of a model's schedule, bound to it: *name*, *form* ("component", "together", "model": a nested model's
    schedule, "method"), its *components* (the components it runs: a method's or a nested model's, all of theirs),
    *stages*, *cadence* (a Cadence or None) and its reads and writes (frozensets of (DataStructure, variable), the
    DataStructure the root one: reads(), writes()).
    """

    def __init__(self, model, name: str, form: str, components, target=None, stages=None, cadence=None):
        self.model, self.name, self.form = model, name, form
        self.components, self.target, self.stages, self.cadence = list(components), target, stages, cadence

    def __repr__(self) -> str:
        return f"ScheduledStep({self.name!r}, {self.form}, {[type(c).__name__ for c in self.components]})"

    @property
    def reads(self) -> frozenset:
        return frozenset(access for component in self.components for access in component_reads(component))

    @property
    def writes(self) -> frozenset:
        return frozenset(access for component in self.components for access in component_writes(component))

    def run(self) -> bool:
        """Run the step (its cadence's due calls only); whether it ran."""
        from openalea.metafspm.coupling.cadence import run_cadenced
        if self.form == "model":
            self.target.run()
            return True
        if self.form == "method":
            self.target()
            return True
        cadence = _effective(self.cadence, self.components[0])
        if self.form == "component" and cadence is None and self.stages is None:
            self.components[0]()                        # its call (Component.__call__: timed)
            self.components[0].__dict__["_ran"] = True
            return True
        started = perf_counter()
        if cadence is None:
            self._call(None)
            due = True
        else:
            due = run_cadenced(self.components, cadence, self._call, key=self.name)
        if self.form == "component":
            component = self.components[0]
            component.__dict__["_wall_time"] = component.__dict__.get("_wall_time", 0.) + perf_counter() - started
        return due

    def _call(self, overrides) -> None:
        if self.form == "component":
            self.components[0]._call(overrides[0] if overrides else None, stages=self.stages)
        else:
            from openalea.metafspm.coupling.composite_wrapper import _run_together
            _run_together(self.components, overrides or [None] * len(self.components))


def _effective(cadence, component):
    """The step's cadence as a Cadence, None when it is every scene step (no cadence, or an interval of the scene's
    step or shorter)."""
    from openalea.metafspm.coupling.cadence import as_cadence
    cadence = as_cadence(cadence)
    if cadence is None:
        return None
    step = component.simulation_step()
    if step is not None and cadence.interval <= float(step) * (1. + 1e-9):
        return None
    return cadence


def bind(model) -> list:
    """The ScheduledSteps of *model*: its schedule's entries (default: its parts, or its components, in order),
    resolved on its attributes, their cadences those its scenario's "cadence" sets over the declared ones."""
    from openalea.metafspm.coupling.cadence import as_cadence, with_interval
    from openalea.metafspm.coupling.composite_wrapper import CompositeModel
    entries = type(model).schedule
    if entries is None:                             # the parts (an environment model: its components) with steps
        parts = getattr(type(model), "parts", None)
        entries = tuple(parts) if parts else tuple(_attribute_of(model, component) for component in model.components)
        entries = tuple(name for name in entries if _has_steps(getattr(model, name, None)))
    overrides = model.__dict__.get("_cadences", {})
    steps, names = [], set()
    for entry in entries:
        entry = Step(entry) if isinstance(entry, str) else entry
        if not isinstance(entry, Step):
            raise TypeError(f"{type(model).__name__}.schedule: part names, Step(...) or Together(...), got {entry!r}")
        if entry.name in names:
            raise ValueError(f"{type(model).__name__}.schedule: two steps named '{entry.name}' (a part scheduled twice "
                             "names its other steps: Step(..., name=...))")
        names.add(entry.name)
        cadence = entry.cadence
        if entry.name in overrides:
            given = overrides[entry.name]
            declared = as_cadence(cadence)
            cadence = with_interval(given, declared.at if declared is not None else None)
        targets = [_resolve(model, target) for target in entry.targets]
        if isinstance(entry, Together):
            if not all(form == "component" for form, _ in targets):
                raise TypeError(f"{type(model).__name__}.schedule: Together runs components, {entry.targets}")
            steps.append(ScheduledStep(model, entry.name, "together", [t for _, t in targets], cadence=cadence))
            continue
        ((form, target),) = targets
        if form == "component":
            steps.append(ScheduledStep(model, entry.name, form, [target], target, entry.stages, cadence))
            continue
        if entry.stages is not None or cadence is not None:
            raise ValueError(f"{type(model).__name__}.schedule: {entry.name}: stages and a cadence are a component's "
                             "step's (or Together's), not a method's or a nested model's")
        components = target.components if form == "model" else model.components
        steps.append(ScheduledStep(model, entry.name, form, components, target))
    unknown = set(overrides) - names
    if unknown:
        raise KeyError(f"{type(model).__name__}: cadence for {sorted(unknown)}: no such step (its steps: "
                       f"{[step.name for step in steps]})")
    return steps


def _has_steps(part) -> bool:
    """Whether a part runs anything: a nested model, or a component whose class has steps (an initiator alone, a
    crown building the plant, has none: not scheduled by default)."""
    from openalea.metafspm.coupling.composite_wrapper import CompositeModel
    if isinstance(part, CompositeModel):
        return True
    schedule = getattr(type(part), "schedule", None)
    return not callable(schedule) or bool(schedule()) or type(part).__call__ is not _base_call()


def _base_call():
    from openalea.metafspm.coupling.component import Component
    return Component.__call__


def _resolve(model, name: str) -> tuple:
    """(form, target) of a schedule entry's name: a component or a nested model of the model's, or its method."""
    from openalea.metafspm.coupling.composite_wrapper import CompositeModel
    target = getattr(model, name, None)
    if isinstance(target, CompositeModel):
        return "model", target
    if target is not None and any(target is component for component in model.components):
        return "component", target
    if callable(target) and getattr(target, "__self__", None) is model:
        return "method", target
    raise KeyError(f"{type(model).__name__}.schedule: '{name}' is none of its parts, components or methods")


def _attribute_of(model, component) -> str:
    """The name of *model*'s attribute holding *component* (an environment model's default schedule)."""
    for name, value in vars(model).items():
        if value is component:
            return name
    raise KeyError(f"{type(model).__name__}: its component {type(component).__name__} is no attribute of it: give "
                   "the model a schedule, or the component an attribute")


# ── reads and writes ─────────────────────────────────────────────────────────

def _roots(data_structure) -> list:
    """The root DataStructures a component's DataStructure holds its variables in (a union's: its parts')."""
    from openalea.metafspm.data_structure.data_api import _root_data_structure
    parts = getattr(data_structure, "parts", None)
    if parts is not None and not hasattr(data_structure, "register"):
        return [_root_data_structure(part) for part in parts]
    return [_root_data_structure(data_structure)]


def _stored(root, name: str, private: dict) -> set:
    """The variables *name* reads on *root*: through its private name, its aliases and its derivations' sources."""
    found, pending, seen = set(), [private.get(name, name)], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        resolved = root.aliases().get(current, current)
        while resolved in root.aliases() and resolved not in seen:
            seen.add(resolved)
            resolved = root.aliases()[resolved]
        if resolved in root.derived():
            pending.extend(str(source) for source in root._derived_sources(resolved))
        else:
            found.add(resolved)
    return found


def component_reads(component) -> set:
    """{(DataStructure, variable)} *component* reads in its calls: its inputs and states (and parameters stored on its
    DataStructure) as stored, the topology of its DataStructure, the plants' geometry for a geometry input."""
    specs = getattr(component, "_variable_specs", {})
    private = component.__dict__.get("_private_names") or {}
    found = set()
    for root in _roots(component.data_structure):
        found.add((root, STRUCTURE))
        for name, spec in specs.items():
            if spec.role in ("input_variable", "state_variable", "parameter") and root.has(private.get(name, name)):
                found.update((root, stored) for stored in _stored(root, name, private))
        if _reads_geometry(component):
            found.add((root, GEOMETRY))
    return found


def component_writes(component) -> set:
    """{(DataStructure, variable)} *component* writes in its calls: its state variables as stored, and the topology
    and the geometry of its DataStructure for a structural component with steps."""
    from openalea.metafspm.coupling.component import StructuralComponent
    specs = getattr(component, "_variable_specs", {})
    found = set()
    for root in _roots(component.data_structure):
        for name, spec in specs.items():
            if spec.role == "state_variable" and root.has(name):
                found.add((root, root.aliases().get(name, name)))
        if isinstance(component, StructuralComponent) and component.schedule():
            found.update({(root, STRUCTURE), (root, GEOMETRY)})
    return found


def _reads_geometry(component) -> bool:
    from dataclasses import fields
    return hasattr(component, "__dataclass_fields__") and any(
        f.metadata.get("role") == "geometry" for f in fields(component))


def conflicts(first: ScheduledStep, second: ScheduledStep) -> frozenset:
    """The accesses two steps share with at least one writing: empty when they may run concurrently."""
    return frozenset((first.writes & (second.reads | second.writes)) | (second.writes & first.reads))
