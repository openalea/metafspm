import inspect as ins
from functools import partial
import sys


# General process resolution method
class Functor:
    def __init__(self, fun, iteraring: bool = False, total: bool = False):
        self.fun = fun
        self.name = self.fun.__name__[1:]
        self.class_name = self.fun.__qualname__.split('.')[0]
        self.iterating = iteraring
        self.total = total
        self.input_names = self.inputs(self.fun)
        if len(self.input_names) == 0:
            self.iterating = True

    def inputs(self, fun):
        arguments = ins.getfullargspec(fun)[0]
        arguments.remove("self")
        return arguments

    def __call__(self, instance, data, data_type="<class 'dict'>", *args):
        if self.iterating:
            self.fun(instance)
        elif data_type == "<class 'dict'>":
            if self.total:
                data[self.name].update(
                    {1: self.fun(instance, *(data[arg] for arg in self.input_names))})
            else:
                # print(self.name, self.input_names)
                # print([getattr(instance, arg) for arg in self.input_names])
                # print(self.name, {arg: data[arg] for arg in self.input_names})
                # if self.name == "mycorrhizal_mediated_import_Nm":
                #     print(self.name, {arg: data[arg] for arg in self.input_names})
                # print(self.name, [arg for arg in self.input_names if 254 not in data[arg].keys()])
                data[self.name].update(
                    {vid: self.fun(instance, *(data[arg][vid] for arg in self.input_names)) for vid in data["focus_elements"]})
                
        elif data_type == "<class 'numpy.ndarray'>":
            data[self.name] = self.fun(instance, *(data[arg] for arg in self.input_names))

# Executor singleton
class Singleton(object):
    _instance = None
    universal_steps = ["priorbalance", "selfbalance", "stepinit", "state", "totalstate", "rate", "totalrate", "deficit", 
                       "axial", "potential", "allocation", "actual", "segmentation", "postsegmentation"]

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
            class_._instance.priorbalance = {}
            class_._instance.selfbalance = {}
            class_._instance.stepinit = {}
            class_._instance.state = {}
            class_._instance.totalstate = {}
            class_._instance.rate = {}
            class_._instance.totalrate = {}
            class_._instance.deficit = {}
            class_._instance.axial = {}
            class_._instance.potential = {}
            class_._instance.allocation = {}
            class_._instance.actual = {}
            class_._instance.segmentation = {}
            class_._instance.postsegmentation = {}

        return class_._instance


class Choregrapher(Singleton):
    """
    This Singleton class retreives the processes tagged by a decorator in a model class.
    It also provides a __call__ method to schedule model execution.
    """

    scheduled_groups = {}
    sub_time_step = {}
    data_structure = {"soil":None, "root":None}
    filter =  {"label": ["Segment", "Apex"], "type":["Base_of_the_root_system", "Normal_root_after_emergence", "Stopped", "Just_Stopped", "Root_nodule"]}

    consensus_scheduling = [
            ["priorbalance", "selfbalance"],
            ["stepinit", "rate", "totalrate", "state", "totalstate"],  # metabolic models
            ["axial"],  # subcategoy for metabolic models
            ["potential", "deficit", "allocation", "actual", "segmentation", "postsegmentation"],  # growth models
        ]


    def add_time_and_data(self, instance, sub_time_step: int, data: dict, compartment: str = "root"):
        # module_family = instance.family
        module_family = instance.__class__.__name__
        self.sub_time_step[module_family] = sub_time_step
        if self.data_structure[compartment] == None:
            self.data_structure[compartment] = data
        data_structure_type = str(type(list(self.data_structure[compartment].values())[0]))
        for k in self.scheduled_groups[module_family].keys():
            for f in range(len(self.scheduled_groups[module_family][k])):
                self.scheduled_groups[module_family][k][f] = partial(self.scheduled_groups[module_family][k][f], *(instance, self.data_structure[compartment], data_structure_type))


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
        if self.data_structure['root'] is not None:
            if "focus_elements" not in self.data_structure["root"].keys():
                self.data_structure["root"]["focus_elements"] = [vid for vid in self.data_structure["root"]["struct_mass"].keys() if (
                    self.data_structure["root"]["label"][vid] in self.filter["label"] 
                    and self.data_structure["root"]["type"][vid] in self.filter["type"])]

        for increment in range(int(self.simulation_time_step/self.sub_time_step[module_family])):
            for step in self.scheduled_groups[module_family].keys():
                for functor in self.scheduled_groups[module_family][step]:
                    functor()

        if module_family.lower().startswith("rootgrowth"):
            self.data_structure["root"]["focus_elements"] = [vid for vid in self.data_structure["root"]["struct_mass"].keys() if (
                self.data_structure["root"]["label"][vid] in self.filter["label"] 
                and self.data_structure["root"]["type"][vid] in self.filter["type"])]

        
# Decorators    
def priorbalance(func):
    def wrapper():
        Choregrapher().add_process(Functor(func, iteraring=True), name="priorbalance")
        return func
    return wrapper()

def selfbalance(func):
    def wrapper():
        Choregrapher().add_process(Functor(func, iteraring=True), name="selfbalance")
        return func
    return wrapper()

def stepinit(func):
    def wrapper():
        Choregrapher().add_process(Functor(func, iteraring=True), name="stepinit")
        return func
    return wrapper()                

def state(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="state")
        return func
    return wrapper()


def rate(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="rate")
        return func
    return wrapper()

def totalrate(func):
    def wrapper():
        Choregrapher().add_process(Functor(func, total=True), name="totalrate")
        return func
    return wrapper()

def deficit(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="deficit")
        return func
    return wrapper()


def totalstate(func):
    def wrapper():
        Choregrapher().add_process(Functor(func, total=True), name="totalstate")
        return func
    return wrapper()

def axial(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="axial")
        return func
    return wrapper()


def potential(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="potential")
        return func
    return wrapper()

def allocation(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="allocation")
        return func
    return wrapper()

def actual(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="actual")
        return func
    return wrapper()


def segmentation(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="segmentation")
        return func
    return wrapper()

def postsegmentation(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="postsegmentation")
        return func
    return wrapper()
