import ast
from functools import partial


# The steps the step decorators register at class definition, {category: {class family: [functors]}}: what a class
# declares (like its fields), not a run's state. The run's state is on the components: their bound schedule
# (Choregrapher.bind), their clock, and the simulation step, their DataStructure's (DataStructure.simulation_time_step,
# the Scene's time step).
UNIVERSAL_STEPS = ("stepinit", "state", "totalstate", "rate", "totalrate", "deficit", "axial", "potential",
                   "allocation", "actual", "segmentation", "postsegmentation")
_PROCESSES = {name: {} for name in UNIVERSAL_STEPS}


class Choregrapher:
    """
    The schedule of the components' steps: it reads the processes tagged by the step decorators in a component class
    (registered at class definition) and runs a component's bound steps in the consensus order. It holds no run state
    (any number of Scenes, of any time steps, run in one process): a component's bound schedule, clock and simulation
    step are its own (the simulation step its DataStructure's, set by the Scene).
    """

    universal_steps = UNIVERSAL_STEPS
    consensus_scheduling = [
            ["stepinit", "rate", "totalrate", "state", "totalstate"],  # metabolic models
            ["axial"],  # subcategoy for metabolic models
            ["potential", "deficit", "allocation", "actual", "segmentation", "postsegmentation"],  # growth models
        ]

    def bind(self, instance, time_step: float, data) -> None:
        """
        Bind the steps of the instance's class, its own and those inherited from its bases, to the instance and
        its DataStructure *data*, run every *time_step* (s). The bound schedule is kept on the instance, so that
        several instances of one class run their own steps on their own DataStructures.
        """
        groups = {priority: [partial(functor, instance, data) for functor in functors]
                  for priority, functors in self.schedule_of(type(instance)).items()}
        instance.__dict__["_choregraphy"] = (time_step, groups)

    def add_process(self, f, name):
        """Register step functor *f* in category *name* for its class, identified by module and qualified name."""
        family = f.family
        registered = _PROCESSES[name].setdefault(family, [])
        for k, other in enumerate(registered):
            if other.name == f.name:        # the class was defined again (e.g. reloaded): the latest definition wins
                registered[k] = f
                return
        registered.append(f)

    def _steps_of_family(self, family) -> dict:
        """{step name: (functor, categories)} declared by one class."""
        steps = {}
        for category in self.universal_steps:
            for functor in _PROCESSES[category].get(family, []):
                steps.setdefault(functor.name, [functor, set()])[1].add(category)
        return steps

    def schedule_of(self, cls) -> dict:
        """
        Unbound schedule of class *cls*: {priority: [functors]}, sorted. A class runs its own steps and those of its
        bases, a step redefined by a subclass replacing its base's (by name, with its own categories), except the
        steps listed in a steps_removed class attribute.
        """
        steps = {}
        for klass in reversed(cls.__mro__):
            steps.update(self._steps_of_family(family_of(klass)))
        # Steps a class removes from its bases: steps_removed = ("name", ...), names without the leading "_"
        removed = {name.lstrip("_") for klass in cls.__mro__ for name in klass.__dict__.get("steps_removed", ())}
        unknown = removed - set(steps)
        if unknown:
            raise ValueError(f"{cls.__name__}.steps_removed: no step named {sorted(unknown)} (steps: {sorted(steps)})")
        steps = {name: step for name, step in steps.items() if name not in removed}
        groups = {}
        for name, (functor, categories) in steps.items():
            priority = [0] * len(self.consensus_scheduling)
            for row, schedule in enumerate(self.consensus_scheduling):
                for process_type, category in enumerate(schedule):
                    if category in categories:
                        priority[row] = process_type + 1
            groups.setdefault(str(priority), []).append(functor)
        return {k: groups[k] for k in sorted(groups)}

    def __call__(self, instance):
        """
        Run the bound schedule of component *instance* once per sub time step of the simulation step (its own
        time_step by default), its clock at the start of each sub-step.
        """
        sub_time_step, groups = instance.__dict__["_choregraphy"]
        start = instance.__dict__.get("_clock", 0.)
        simulation_step = instance.simulation_step() or sub_time_step
        increments = 1 if sub_time_step is None else sub_steps(simulation_step, sub_time_step, type(instance).__name__)
        # the call's whole step, forcing(over="step")'s
        instance.__dict__["_step_clock"], instance.__dict__["_step_length"] = start, simulation_step
        for increment in range(increments):
            instance.__dict__["_clock"] = start + increment * (sub_time_step or 0.)
            for step in groups:
                for functor in groups[step]:
                    functor()
        instance.__dict__["_clock"] = start


    def run_together(self, instances) -> None:
        """
        Run the bound schedules of several components as one: their step groups by schedule position (e.g. one's
        potential, the other's segmentation, then both postsegmentations), the components in the given order within a
        position; once per sub time step, which they must share.
        """
        names = "+".join(type(instance).__name__ for instance in instances)
        sub_time_steps = {instance.__dict__["_choregraphy"][0] for instance in instances}
        if len(sub_time_steps) > 1:
            raise ValueError(f"{names}: components run together need the same sub time step, got {sub_time_steps}")
        sub_time_step = sub_time_steps.pop()
        simulation_steps = {instance.simulation_step() for instance in instances}
        if len(simulation_steps) > 1:
            raise ValueError(f"{names}: components run together need the same simulation step, got {simulation_steps}")
        simulation_step = simulation_steps.pop() or sub_time_step
        increments = 1 if sub_time_step is None else sub_steps(simulation_step, sub_time_step, names)
        starts = [instance.__dict__.get("_clock", 0.) for instance in instances]
        for instance, start in zip(instances, starts):                                 # forcing(over="step")
            instance.__dict__["_step_clock"], instance.__dict__["_step_length"] = start, simulation_step
        priorities = sorted({priority for instance in instances for priority in instance.__dict__["_choregraphy"][1]},
                            key=lambda priority: tuple(ast.literal_eval(priority)))
        for increment in range(increments):
            for instance, start in zip(instances, starts):
                instance.__dict__["_clock"] = start + increment * (sub_time_step or 0.)
            for priority in priorities:
                for instance in instances:
                    for functor in instance.__dict__["_choregraphy"][1].get(priority, ()):
                        functor()
        for instance, start in zip(instances, starts):
            instance.__dict__["_clock"] = start

    def describe(self, cls) -> list:
        """[(stages, [step names])] of class *cls*, in the order its steps run: each group of steps with the stage it
        has in each row of the schedule (e.g. ("rate",) or ("postsegmentation",))."""
        described = []
        for priority, functors in self.schedule_of(cls).items():
            positions = ast.literal_eval(priority)
            stages = tuple(self.consensus_scheduling[row][position - 1] for row, position in enumerate(positions)
                           if position > 0)
            described.append((stages, [functor.fun.__name__ for functor in functors]))
        return described

    def run_stage(self, instance, stage: str):
        """Run once the steps of component *instance* in the schedule row of *stage* ("potential", "segmentation",
        "postsegmentation", "rate", ...), in their order: e.g. to update a structural component's postsegmentation
        outputs after another model wrote its inputs."""
        rows = [(row, schedule.index(stage) + 1) for row, schedule in enumerate(self.consensus_scheduling)
                if stage in schedule]
        if not rows:
            raise ValueError(f"no schedule row has the stage '{stage}'")
        row, position = rows[0]
        _, groups = instance.__dict__["_choregraphy"]
        for priority, functors in groups.items():
            if ast.literal_eval(priority)[row] == position:
                for functor in functors:
                    functor()


def family_of(cls) -> str:
    """Key of a component class's steps: module and qualified name, so that same-named classes do not collide."""
    return f"{cls.__module__}:{cls.__qualname__}"


def sub_steps(simulation_time_step: float, sub_time_step: float, owner: str = "component") -> int:
    """Number of sub-steps of *sub_time_step* in a simulation step; refused unless a whole number of at least 1."""
    ratio = float(simulation_time_step) / float(sub_time_step)
    count = int(round(ratio))
    if count < 1 or abs(ratio - count) > 1e-9 * max(ratio, 1.):
        raise ValueError(f"{owner}: its time step ({sub_time_step} s) must divide the simulation time step "
                         f"({simulation_time_step} s) a whole number of times")
    return count

