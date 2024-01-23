import inspect as ins
from functools import partial


# General process resolution method
class Functor:
    def __init__(self, fun, iteraring: bool = False, total: bool = False):
        self.fun = fun
        self.name = self.fun.__name__[1:]
        self.iterating = iteraring
        self.total = total
        self.input_names = self.inputs(self.fun)
        if len(self.input_names) == 0:
            self.iterating = True

    def inputs(self, fun):
        arguments = ins.getfullargspec(fun)[0]
        arguments.remove("self")
        return arguments

    def __call__(self, instance, data):
        if self.iterating:
            self.fun(instance)
        elif self.total:
            data[self.name].update(
                {1: self.fun(instance, *(getattr(instance, arg) for arg in self.input_names))})
        else:
            data[self.name].update(
                {vid: self.fun(instance, *(getattr(instance, arg)[vid] for arg in self.input_names)) for vid in data["focus_elements"]})


# Executor singleton
class Singleton(object):
    _instance = []

    def __new__(class_, new_instance = True, **kwargs):
        if new_instance:
            class_._instance.append(object.__new__(class_, **kwargs))
            # We initiate the decorators lists to agregate methods
            class_._instance[-1].priorbalance = []
            class_._instance[-1].selfbalance = []
            class_._instance[-1].postgrowth = []
            class_._instance[-1].stepinit = []
            class_._instance[-1].state = []
            class_._instance[-1].totalstate = []
            class_._instance[-1].rate = []
            class_._instance[-1].totalrate = []
            class_._instance[-1].deficit = []
            class_._instance[-1].axial = []
            class_._instance[-1].potential = []
            class_._instance[-1].allocation = []
            class_._instance[-1].actual = []
            class_._instance[-1].segmentation = []
            class_._instance[-1].postsegmentation = []
        return class_._instance[-1]


class Choregrapher(Singleton):
    """
    This Singleton class retreives the processes tagged by a decorator in a model class.
    It also provides a __call__ method to schedule model execution.
    """

    consensus_scheduling = [
            ["priorbalance", "selfbalance"],
            ["rate", "totalrate", "state", "totalstate"],  # metabolic models
            ["axial"],  # subcategoy for metabolic models
            ["potential", "deficit", "allocation", "actual", "segmentation", "postsegmentation"],  # growth models
            # Note : this has to be placed at the end to held the first places in time step
            ["postgrowth", "stepinit"],  # General time step priority 
        ]

    def add_data(self, instance, data_name: str, filter: dict = {"label":[""], "type":[""]}):
        self.data_structure = getattr(instance, data_name)
        self.filter = filter
        for k in self.scheduled_groups.keys():
            for f in range(len(self.scheduled_groups[k])):
                self.scheduled_groups[k][f] = partial(self.scheduled_groups[k][f], *(instance, self.data_structure))

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
        getattr(self, name).append(f)
        self.build_schedule()

    def build_schedule(self):
        self.scheduled_groups = {}
        # As functors can belong two multiple categories, we store unique names to avoid duplicated instances
        unique_functors = {}
        for attribute in dir(self):
            if not callable(getattr(self, attribute)) and "_" not in attribute:
                for functor in getattr(self, attribute):
                    if functor.name not in unique_functors.keys():
                        unique_functors[functor.name] = functor
        # Then, We go through these unique functors
        for name, functor in unique_functors.items():
            priority = [0 for k in range(len(self.consensus_scheduling))]
            # We go through each row of the consensus scheduling, in order of priority
            for schedule in range(len(self.consensus_scheduling)):
                # We attribute a number in the functor's tuple to provided decorator.
                for process_type in range(len(self.consensus_scheduling[schedule])):
                    if name in [f.name for f in getattr(self, self.consensus_scheduling[schedule][process_type])]:
                        priority[schedule] = process_type
            # We append the priority tuple to she scheduled groups dictionnary
            if str(priority) not in self.scheduled_groups.keys():
                self.scheduled_groups[str(priority)] = []
            self.scheduled_groups[str(priority)].append(functor)

        # Finally, we sort the dictionnary by key so that the call function can go through functor groups in the expected order
        self.scheduled_groups = {k: self.scheduled_groups[k] for k in sorted(self.scheduled_groups.keys())}

    def __call__(self):
        self.data_structure["focus_elements"] = [vid for vid in self.data_structure["struct_mass"].keys() if (
            self.data_structure["label"][vid] in self.filter["label"] 
            and self.data_structure["type"][vid] in self.filter["type"])]
        for step in self.scheduled_groups.keys():
            for functor in self.scheduled_groups[step]:
                functor()


# Decorators    
def priorbalance(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func, iteraring=True), name="priorbalance")
        return func
    return wrapper()

def selfbalance(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func, iteraring=True), name="selfbalance")
        return func
    return wrapper()

def postgrowth(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func, iteraring=True), name="postgrowth")
        return func
    return wrapper()

def stepinit(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func, iteraring=True), name="stepinit")
        return func
    return wrapper()                

def state(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func), name="state")
        return func
    return wrapper()


def rate(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func), name="rate")
        return func
    return wrapper()

def totalrate(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func, total=True), name="totalrate")
        return func
    return wrapper()

def deficit(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func), name="deficit")
        return func
    return wrapper()


def totalstate(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func, total=True), name="totalstate")
        return func
    return wrapper()

def axial(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func), name="axial")
        return func
    return wrapper()


def potential(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func), name="potential")
        return func
    return wrapper()

def allocation(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func), name="allocation")
        return func
    return wrapper()

def actual(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func), name="actual")
        return func
    return wrapper()


def segmentation(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func), name="segmentation")
        return func
    return wrapper()

def postsegmentation(func):
    def wrapper():
        Choregrapher(new_instance=False).add_process(Functor(func), name="postsegmentation")
        return func
    return wrapper()
