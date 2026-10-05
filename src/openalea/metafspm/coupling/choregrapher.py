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


    def add_schedule(self, schedule):
        """
        Method to edit standarded scheduling proposed by the choregrapher. 
        Guidelines :
        - Rows' index in the list are priority order.
        - Elements' index in the rows are in priority order.
        Thus, you should design the priority of this schedule so that "actual rate" comming before "potential state" is indeed the expected behavior in computation scheduling.
        :param schedule: List of lists of stings associated to available decorators :

        For metabolic models, soil models (priority order) : 
        - rate : for process rate computation that will affect model states (ex : transport flow, metabolic consumption) 
        - state : for state balance computation, from previous state and integration of rates' modification (ex : concentrations and contents)
        - deficit : for abnormal state values resulting from rate balance, cumulative deficits are computed before thresholding state values (ex : negative concentrations)

        For growth models (priority order) : 
        - potential : potential element growth computations regarding element initial state
        - actual : actual element growth computations regarding element states actualizing structural states (belongs to state)
        - segmentation : single element partitionning in several uppon actual growth if size exceeds a threshold.
        """
        self.consensus_scheduling = schedule


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

    def build_schedule(self, module_family):
        """Former name-keyed schedule builder, kept for code calling it with a class (see schedule_of)."""
        if isinstance(module_family, type):
            self.scheduled_groups[family_of(module_family)] = self.schedule_of(module_family)

    def __call__(self, module_family=None, instance=None):
        """Run the bound schedule of *instance* (or, formerly, of the last bound instance of class *module_family*)."""
        if instance is not None:
            sub_time_step, groups = instance.__dict__["_choregraphy"]
        else:
            family = next((f for f in self.scheduled_groups if f == module_family or f.endswith(f":{module_family}")),
                          module_family)
            sub_time_step, groups = self.sub_time_step[family], self.scheduled_groups[family]
        for increment in range(int(self.simulation_time_step / sub_time_step)):
            for step in groups:
                for functor in groups[step]:
                    functor()


def family_of(cls) -> str:
    """Key of a component class's steps: module and qualified name, so that same-named classes do not collide."""
    return f"{cls.__module__}:{cls.__qualname__}"
