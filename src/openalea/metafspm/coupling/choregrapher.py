from functools import partial


# Executor singleton
class Singleton:
    _instance = None
    universal_steps = ["priorbalance", "selfbalance", "stepinit", "state", "totalstate", "rate", "totalrate", "deficit", 
                       "axial", "potential", "allocation", "actual", "segmentation", "postsegmentation"]

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            inst = object.__new__(class_, *args, **kwargs)
            inst._init_state()
            class_._instance = inst

        return class_._instance
    
    def _init_state(self):
        # centralize state initialization
        for name in self.universal_steps:
            setattr(self, name, {})
    

class Choregrapher(Singleton):
    """
    This Singleton class retreives the processes tagged by a decorator in a model class.
    It also provides a __call__ method to schedule model execution.
    """

    consensus_scheduling = [
            ["priorbalance", "selfbalance"],
            ["stepinit", "rate", "totalrate", "state", "totalstate"],  # metabolic models
            ["axial"],  # subcategoy for metabolic models
            ["potential", "deficit", "allocation", "actual", "segmentation", "postsegmentation"],  # growth models
        ]


    def _init_state(self):
        super()._init_state()
        self.scheduled_groups = {}
        self.sub_time_step = {}
        self.data_structure = {}


    def reset(self):
        """
        Clear run state (bound schedules, sub time steps, data structures, simulation time step) in place.

        The processes registered by step decorators at class definition are kept, so that component classes
        defined before the reset can still be instantiated and scheduled. The instance is kept as well, since
        Component.choregrapher is bound at class creation and a new instance would never be called.
        """
        self.scheduled_groups = {}
        self.sub_time_step = {}
        self.data_structure = {}
        if "simulation_time_step" in self.__dict__:
            del self.simulation_time_step


    def add_time_and_data(self, instance, sub_time_step: int, data, compartment: str = "graph"):
        """
        Bind the steps of the instance's class, its own and those inherited from its bases, to the instance and
        its DataStructure. The bound schedule is kept on the instance, so that several instances of one class run
        their own steps on their own DataStructures.

        Args:
            instance: component instance whose class the steps were collected from
            sub_time_step (int): sub time step of the component (the simulation step is divided accordingly)
            data: the component's DataStructure
            compartment (str, optional): name under which the DataStructure is recorded
        """
        family = family_of(type(instance))
        self.sub_time_step[family] = sub_time_step
        self.data_structure[compartment] = data
        groups = {priority: [partial(functor, instance, data) for functor in functors]
                  for priority, functors in self.schedule_of(type(instance)).items()}
        instance.__dict__["_choregraphy"] = (sub_time_step, groups)
        self.scheduled_groups[family] = groups      # introspection: the last bound instance of the class

    def add_simulation_time_step(self, simulation_time_step: int):
        """
        Enables to add a global simulation time step to the Choregrapher for it to slice subtimesteps accordingly
        :param simulation_time_step: global simulation time step in seconds
        :return:
        """
        self.simulation_time_step = simulation_time_step


    def add_process(self, f, name):
        """Register step functor *f* in category *name* for its class, identified by module and qualified name."""
        family = f.family
        registered = getattr(self, name).setdefault(family, [])
        for k, other in enumerate(registered):
            if other.name == f.name:        # the class was defined again (e.g. reloaded): the latest definition wins
                registered[k] = f
                return
        registered.append(f)

    def _steps_of_family(self, family) -> dict:
        """{step name: (functor, categories)} declared by one class."""
        steps = {}
        for category in self.universal_steps:
            for functor in getattr(self, category).get(family, []):
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
        for increment in range(sub_steps(self.simulation_time_step, sub_time_step, type(instance).__name__)):
            instance.__dict__["_clock"] = start + increment * sub_time_step
            for step in groups:
                for functor in groups[step]:
                    functor()
        instance.__dict__["_clock"] = start


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

