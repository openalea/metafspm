import inspect as ins
from functools import partial
from dataclasses import dataclass


# General process resolution method
class Functor:
    def __init__(self, fun):
        self.fun = fun
        self.name = self.fun.__name__
        self.input_names = self.inputs(self.fun)

    def inputs(self, fun):
        arguments = ins.getfullargspec(fun)[0]
        arguments.remove("self")
        return arguments

    def __call__(self, data):
        data[self.name].update(
            {vid: self.fun(self, *(data[arg][vid] for arg in self.input_names)) for vid in data["emerged_elements"]})


# Executor singleton
class Singleton(object):
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance


class Executor(Singleton):
    state = []
    rate = []
    potential = []
    actual = []
    segmentation = []

    consensus_scheduling = [
        ["rate", "state"],  # metabolic models
        ["potential", "actual", "segmentation"]  # growth models
    ]

    def add_data(self, data):
        self.data_structure = data
        for k in self.scheduled_groups.keys():
            for f in range(len(self.scheduled_groups[k])):
                self.scheduled_groups[k][f] = partial(self.scheduled_groups[k][f], self.data_structure)

    def add_process(self, f, name):
        getattr(self, name).append(f)
        self.build_schedule()

    def build_schedule(self):

        self.scheduled_groups = {}

        unique_functors = {}
        for attribute in dir(self):
            if not callable(getattr(self, attribute)) and "_" not in attribute:
                for functor in getattr(self, attribute):
                    if functor.name not in unique_functors.keys():
                        unique_functors[functor.name] = functor

        for name, functor in unique_functors.items():
            priority = [0 for k in range(len(self.consensus_scheduling))]
            for schedule in range(len(self.consensus_scheduling)):
                for process_type in range(len(self.consensus_scheduling[schedule])):
                    if name in [f.name for f in getattr(self, self.consensus_scheduling[schedule][process_type])]:
                        priority[schedule] = process_type
            if str(priority) not in self.scheduled_groups.keys():
                self.scheduled_groups[str(priority)] = []
            self.scheduled_groups[str(priority)].append(functor)

        self.scheduled_groups = {k: self.scheduled_groups[k] for k in sorted(self.scheduled_groups.keys())}

    def __call__(self):
        for step in self.scheduled_groups.keys():
            for functor in self.scheduled_groups[step]:
                functor()


# Decorators
def state(func):
    def wrapper():
        Executor().add_process(Functor(func), name="state")
        return func

    return wrapper()


def rate(func):
    def wrapper():
        Executor().add_process(Functor(func), name="rate")
        return func

    return wrapper()


def potential(func):
    def wrapper():
        Executor().add_process(Functor(func), name="potential")
        return func

    return wrapper()


def actual(func):
    def wrapper():
        Executor().add_process(Functor(func), name="actual")
        return func

    return wrapper()


def segmentation(func):
    def wrapper():
        Executor().add_process(Functor(func), name="segmentation")
        return func

    return wrapper()



# Only demo here
# Actual base component wich can use decorators
@dataclass
class Component:
    executor = Executor()

    def __call__(self):
        self.executor()

    # + d'autres méthodes génériques bien sûr, injection de scenario, initialisations, etc


### Modeler side ###
class Model(Component):

    # constrained field initialization

    def __init__(self, g_properties):
        self.props = g_properties
        self.executor.add_data(data=self.props)

    @rate
    def hexose_exudation(self, hexose):
        print("First")
        return 1.

    @rate
    def sucrose_unloading(self, sucrose, hexose):
        print("Second")
        return 1.

    @actual
    @state
    def hexose(self, hexose, hexose_exudation):
        print("Third")
        return 1.

    @potential
    @state
    def sucrose(self, sucrose, sucrose_unloading):
        print("Fourth")
        return 1.


if __name__ == "__main__":
    g_properties = {"emerged_elements": ["1", "2", "3"],
                    "hexose": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "sucrose": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "hexose_exudation": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "sucrose_unloading": {"1": 0., "2": 0., "3": 0., "4": 0.}}
    model = Model(g_properties)
    print("before", g_properties)
    for step in range(1):
        model()
    print("after", g_properties)

