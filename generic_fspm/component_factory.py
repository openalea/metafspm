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
    _instance = []

    def __new__(class_, new_instance = True, **kwargs):
        if new_instance:
            class_._instance.append(object.__new__(class_, **kwargs))
            class_._instance[-1].state = []
            class_._instance[-1].rate = []
            class_._instance[-1].deficit = []
            class_._instance[-1].potential = []
            class_._instance[-1].actual = []
            class_._instance[-1].segmentation = []
        return class_._instance[-1]


class Executor(Singleton):
    """
    This Singleton class retreives the processes tagged by a decorator in a model class.
    It also provides a __call__ method to schedule model execution.
    """

    consensus_scheduling = [
            ["rate", "state", "deficit"],  # metabolic models
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
        for step in self.scheduled_groups.keys():
            for functor in self.scheduled_groups[step]:
                functor()


# Decorators
def state(func):
    def wrapper():
        Executor(new_instance=False).add_process(Functor(func), name="state")
        return func
    return wrapper()


def rate(func):
    def wrapper():
        Executor(new_instance=False).add_process(Functor(func), name="rate")
        return func
    return wrapper()

def deficit(func):
    def wrapper():
        Executor(new_instance=False).add_process(Functor(func), name="deficit")
        return func
    return wrapper()


def potential(func):
    def wrapper():
        Executor(new_instance=False).add_process(Functor(func), name="potential")
        return func
    return wrapper()


def actual(func):
    def wrapper():
        Executor(new_instance=False).add_process(Functor(func), name="actual")
        return func
    return wrapper()


def segmentation(func):
    def wrapper():
        Executor(new_instance=False).add_process(Functor(func), name="segmentation")
        return func
    return wrapper()


# Only demo here
# Actual base component wich can use decorators
# @dataclass
# class Component:
#     executor = Executor()

#     def __call__(self):
#         self.executor()

#     # + d'autres méthodes génériques bien sûr, injection de scenario, initialisations, etc





