"""
Per-instance scheduling: steps are registered per class (module and
qualified name), collected through the class's bases, and bound per instance, so that several instances of one class
run on their own DataStructures in any order, same-named classes of different modules do not collide, and subclasses
run their bases' steps.
"""
import types
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.solve.decorator import rate

DOC = dict(unit="adim", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])


@dataclass
class Increment(FunctionalComponent):
    level: float = state_variable(**DOC, default=0., location="cell")

    @rate
    def _level(self, level):
        return level + 1.


@dataclass
class InheritsIncrement(Increment):
    pass


@dataclass
class DoublesIncrement(Increment):
    @rate
    def _level(self, level):
        return level + 2.


def _grid():
    return ArrayDataStructure(shape=(2,))


def test_two_instances_of_one_class_run_on_their_own_data_structures():
    a, b = _grid(), _grid()
    first, second = Increment(data_structure=a), Increment(data_structure=b)
    first()
    first()
    second()
    first()
    np.testing.assert_array_equal(a.get("level"), 3.)
    np.testing.assert_array_equal(b.get("level"), 1.)


def test_subclasses_run_their_bases_steps_or_their_own():
    inherited, doubled = _grid(), _grid()
    InheritsIncrement(data_structure=inherited)()
    DoublesIncrement(data_structure=doubled)()
    np.testing.assert_array_equal(inherited.get("level"), 1.)       # formerly no step at all
    np.testing.assert_array_equal(doubled.get("level"), 2.)         # the subclass's step replaces its base's


SOURCE = '''
from dataclasses import dataclass
from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.solve.decorator import rate

@dataclass
class Decay(FunctionalComponent):
    level: float = state_variable(unit="", unit_comment="", description="", min_value=0., max_value=1.,
                                  value_comment="", references="", DOI=[], default=1., location="cell")

    @rate
    def _level(self, level):
        return FACTOR * level
'''


def _module(name, factor):
    module = types.ModuleType(name)
    module.FACTOR = factor
    exec(compile(SOURCE, name, "exec"), module.__dict__)
    return module


def test_same_named_classes_of_different_modules_do_not_collide():
    half, tenth = _module("decay_half", 0.5).Decay, _module("decay_tenth", 0.1).Decay
    a, b = _grid(), _grid()
    half(data_structure=a)()
    tenth(data_structure=b)()
    np.testing.assert_allclose(a.get("level"), 0.5)
    np.testing.assert_allclose(b.get("level"), 0.1)


def test_classes_defined_in_a_function_are_scheduled():
    @dataclass
    class Local(FunctionalComponent):
        level: float = state_variable(**DOC, default=0., location="cell")

        @rate
        def _level(self, level):
            return level + 5.

    data_structure = _grid()
    Local(data_structure=data_structure)()
    np.testing.assert_array_equal(data_structure.get("level"), 5.)
