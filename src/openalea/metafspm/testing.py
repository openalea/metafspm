"""
Checks for downstream packages (design note WD.8): run them in a package's own tests to make sure its components
stay couplable with the translators it ships.
"""
from dataclasses import fields

# Framework fields that are not model variables
_FRAMEWORK_FIELDS = {"data_structure"}


def couplability_problems(component_cls, translator, name: str = None) -> list:
    """
    Problems preventing *component_cls* from being coupled through *translator* (a coupling.translator.Translator):
      * a declared field without declare() metadata;
      * a link whose receiving variable the component does not declare;
      * a link reading a variable of the component that it does not declare.
    *name* is the component name used in the translator (default: the class name).
    """
    name = name or component_cls.__name__
    declared = {f.name: f for f in fields(component_cls) if f.name not in _FRAMEWORK_FIELDS}
    problems = [f"{name}.{f.name} has no declare() metadata (variable_type, by, unit, ...)"
                for f in declared.values() if "variable_type" not in f.metadata]
    for link in translator.links:
        if link.receiver == name and link.variable not in declared:
            problems.append(f"{name}.{link.variable} is linked from {link.provider} but {name} does not declare it")
        if link.provider == name:
            for source in link.sources:
                if source not in declared:
                    problems.append(f"{link.receiver}.{link.variable} reads {name}.{source}, which {name} does not declare")
    return problems


def assert_component_couplable(component_cls, translator, name: str = None) -> None:
    problems = couplability_problems(component_cls, translator, name=name)
    if problems:
        raise AssertionError(f"{name or component_cls.__name__} is not couplable:\n  " + "\n  ".join(problems))
