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
        Bind the steps collected for the instance's class to the instance and its DataStructure.

        Args:
            instance: component instance whose class the steps were collected from
            sub_time_step (int): sub time step of the component (the simulation step is divided accordingly)
            data: the component's DataStructure
            compartment (str, optional): name under which the DataStructure is recorded
        """
        module_family = instance.__class__.__name__
        self.sub_time_step[module_family] = sub_time_step
        self.data_structure[compartment] = data
        self.build_schedule(module_family)
        for k in self.scheduled_groups[module_family].keys():
            for f in range(len(self.scheduled_groups[module_family][k])):
                functor = self.scheduled_groups[module_family][k][f]
                self.scheduled_groups[module_family][k][f] = partial(functor, instance, data)


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
        module_family = f.class_name

        class_globals = f.fun.__globals__
        if "inheriting" in class_globals:
            parent_names = [cls.__name__ for cls in class_globals["inheriting"] if cls.__name__ not in ("object", "Model")]
            # We check all step to transfer them to module familly instead of their base class
            for step in self.universal_steps:
                for parent in parent_names:
                    if parent in getattr(self, step).keys():
                        # In case this is the very first process
                        if module_family not in getattr(self, step).keys():
                            getattr(self, step)[module_family] = []

                        # Gather all the processes from the parent
                        # NOTE : normally the bellow code would replace same names by children's process, as expected by inheritance
                        for process in getattr(self, step)[parent]:
                             getattr(self, step)[module_family].append(process)
                        # Remove parents from the registered modules
                        del getattr(self, step)[parent]

        exists = False
        if module_family not in getattr(self, name).keys():
            getattr(self, name)[module_family] = []
        else:
            for k in range(len(getattr(self, name)[module_family])):
                # If current function already has been flagged, it is replaced cause we suppose that execution order reflects inheritance from parent to children
                # So override is the expected behavior
                f_name = getattr(self, name)[module_family][k].name
                if f_name == f.name:
                    getattr(self, name)[module_family][k] = f
                    exists = True
        if not exists:
            getattr(self, name)[module_family].append(f)
        self.build_schedule(module_family=module_family)


    def build_schedule(self, module_family):
        self.scheduled_groups[module_family] = {}
        # As functors can belong two multiple categories, we store unique names to avoid duplicated instances
        unique_functors = {}
        for attribute in dir(self):
            if not callable(getattr(self, attribute)) and "_" not in attribute:
                if module_family in getattr(self, attribute).keys():
                    for functor in getattr(self, attribute)[module_family]:
                        if functor.name not in unique_functors.keys():
                            unique_functors[functor.name] = functor
        # Then, We go through these unique functors
        for name, functor in unique_functors.items():
            priority = [0 for k in range(len(self.consensus_scheduling))]
            # We go through each row of the consensus scheduling, in order of priority
            for schedule in range(len(self.consensus_scheduling)):
                # We attribute a number in the functor's tuple to provided decorator.
                for process_type in range(len(self.consensus_scheduling[schedule])):
                    considered_step = getattr(self, self.consensus_scheduling[schedule][process_type])
                    if module_family in considered_step.keys():
                        if name in [f.name for f in considered_step[module_family]]:
                            priority[schedule] = process_type + 1
                            
            # We append the priority tuple to she scheduled groups dictionnary
            if str(priority) not in self.scheduled_groups[module_family].keys():
                self.scheduled_groups[module_family][str(priority)] = []
            self.scheduled_groups[module_family][str(priority)].append(functor)

        # Finally, we sort the dictionnary by key so that the call function can go through functor groups in the expected order
        self.scheduled_groups[module_family] = {k: self.scheduled_groups[module_family][k] for k in sorted(self.scheduled_groups[module_family].keys())}


    def __call__(self, module_family):
        for increment in range(int(self.simulation_time_step/self.sub_time_step[module_family])):
            for step in self.scheduled_groups[module_family].keys():
                for functor in self.scheduled_groups[module_family][step]:
                    functor()
