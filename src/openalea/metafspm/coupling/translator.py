"""
Coupling translators as Python objects.

A translator is a list of Links: ``receiver.variable <- Σ factor * provider.source`` (or a formula), optionally
changing scale with an aggregation. Python translators can use live references (``scales.SubOrgan``) and
callables; YAML translators (the historical ``receiver: provider: variable: {source: factor}`` files) load into
the same objects, their string factors being parsed by a restricted arithmetic parser instead of ``eval``.
"""
import ast
import importlib.util
import operator
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Union

import yaml

from openalea.metafspm.data_structure.configs import ScalesConfig

Factor = Union[float, int, str]

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def parse_factor(factor: Factor) -> float:
    """Numeric value of a link factor: a number, or an arithmetic expression such as "-12 * 1e6 * 3600"."""
    if isinstance(factor, (int, float)) and not isinstance(factor, bool):
        return float(factor)
    if not isinstance(factor, str):
        raise ValueError(f"Invalid link factor {factor!r}")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](evaluate(node.operand))
        raise ValueError(f"Invalid link factor {factor!r}: only numbers and + - * / ** are allowed")

    try:
        return float(evaluate(ast.parse(factor, mode="eval")))
    except SyntaxError:
        raise ValueError(f"Invalid link factor {factor!r}") from None


def _scale(value) -> Optional[int]:
    """Scale given as a ScalesConfig integer or name."""
    if value is None or isinstance(value, int):
        return value
    scale = getattr(ScalesConfig, str(value), None)
    if not isinstance(scale, int):
        raise ValueError(f"Unknown scale '{value}', expected one of the ScalesConfig names")
    return scale


@dataclass(frozen=True)
class Link:
    """
    ``receiver.variable`` is provided by ``provider``::

        sources:     {provider variable: factor} (weighted sum; factors may be arithmetic strings), or the provider
                     variable names passed to *formula*.
        aggregation: how values are mapped across locations ("sum", "mean", "weighted_mean", "broadcast",
                     "child", "parent", ...), with *weight* for "weighted_mean". Without it, a link between two
                     locations is mapped from its provider's state_variable_type.
        scale, source_scale:
                     the receiver-side (and sources') scale when the link changes scale (a ScalesConfig reference
                     or name); when given, the receiver's (and the sources') declared location must be that
                     scale's.
        target:      a mask of the DataStructure: the mapped values go to its entities only, the others getting the
                     receiver's default (e.g. a segment concentration broadcast to its symplastic Compartments).
    """
    receiver: str
    variable: str
    provider: str
    sources: Union[Mapping[str, Factor], tuple] = field(default_factory=dict)
    scale: Optional[int] = None
    source_scale: Optional[int] = None
    aggregation: Optional[Union[str, Callable]] = None
    weight: Optional[str] = None
    target: Optional[str] = None
    formula: Optional[Callable] = None
    raw_factors: Mapping = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if self.formula is None:
            if not isinstance(self.sources, Mapping) or not self.sources:
                raise ValueError(f"Link {self.receiver}.{self.variable}: sources must be a non-empty {{name: factor}}")
            raw = dict(self.sources)
            object.__setattr__(self, "raw_factors", raw)
            object.__setattr__(self, "sources", {name: parse_factor(f) for name, f in raw.items()})
        else:
            object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "scale", _scale(self.scale))
        object.__setattr__(self, "source_scale", _scale(self.source_scale))

    @property
    def detail(self) -> str:
        """identity | alias | factor | expression | multi_source | same_name_factor | formula | scale_change."""
        if self.formula is not None:
            return "formula"
        if (self.scale is not None or self.source_scale is not None or self.aggregation is not None
                or self.target is not None):
            return "scale_change"
        if len(self.sources) > 1:
            return "multi_source"
        (source, factor), = self.sources.items()
        if isinstance(self.raw_factors.get(source), str):
            return "expression"
        if factor == 1.:
            return "identity" if source == self.variable else "alias"
        return "same_name_factor" if source == self.variable else "factor"

    @property
    def kind(self) -> str:
        """identity (no link needed on one DataStructure) | alias (name-level) | derived (computed)."""
        detail = self.detail
        return detail if detail in ("identity", "alias") else "derived"


@dataclass
class Translator:
    links: list = field(default_factory=list)

    # ── building ──────────────────────────────────────────────────────────────

    def link(self, receiver: str, variable: str, provider: str, sources=None, **options) -> "Translator":
        """Add a Link and return the translator, for chained declarations."""
        self.links.append(Link(receiver, variable, provider, sources if sources is not None else {}, **options))
        return self

    def links_of(self, receiver: str, provider: str = None) -> list:
        return [link for link in self.links if link.receiver == receiver and (provider is None or link.provider == provider)]

    @property
    def components(self) -> list:
        names = []
        for link in self.links:
            for name in (link.receiver, link.provider):
                if name not in names:
                    names.append(name)
        return names

    # ── queries ───────────────────────────────────────────────────────────────

    def inputs_outputs(self, components, target: str, names_for_others: bool = True) -> tuple:
        """
        Variables exchanged between *target* and *components* (same semantics as
        CompositeModel.get_component_inputs_outputs): outputs of *target* read by the components and inputs of
        *target* from them, named on the components' side (names_for_others=True) or on the target's side.
        """
        inputs, outputs = set(), set()
        for component in components:
            if component == target:
                continue
            for link in self.links_of(component, provider=target):
                outputs.update([link.variable] if names_for_others else list(link.sources))
            for link in self.links_of(target, provider=component):
                inputs.update(list(link.sources) if names_for_others else [link.variable])
        return list(inputs), list(outputs)

    # ── conversions ───────────────────────────────────────────────────────────

    def to_nested(self) -> dict:
        """Historical nested format {receiver: {provider: {variable: {source: factor}}}}, every pair present."""
        names = self.components
        nested = {receiver: {provider: {} for provider in names} for receiver in names}
        for link in self.links:
            if link.formula is not None or link.detail == "scale_change":
                raise ValueError(f"Link {link.receiver}.{link.variable} uses a formula or a scale change, "
                                 "which the nested format cannot express")
            nested[link.receiver][link.provider][link.variable] = dict(link.sources)
        return nested

    @classmethod
    def from_dict(cls, nested: dict) -> "Translator":
        translator = cls()
        for receiver, providers in nested.items():
            for provider, links in (providers or {}).items():
                for variable, spec in (links or {}).items():
                    if isinstance(spec, Mapping) and "sources" in spec:
                        options = {key: spec[key] for key in ("scale", "source_scale", "aggregation", "weight", "target")
                                   if key in spec}
                        translator.link(receiver, variable, provider, dict(spec["sources"]), **options)
                    else:
                        translator.link(receiver, variable, provider, dict(spec))
        return translator

    @classmethod
    def from_yaml(cls, path) -> "Translator":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_module(cls, path_or_module) -> "Translator":
        """Translator exposed as ``translator`` by a Python module (file path or imported module)."""
        module = path_or_module
        if isinstance(path_or_module, str):
            spec = importlib.util.spec_from_file_location("_metafspm_coupling_translator", path_or_module)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        translator = getattr(module, "translator", None)
        if not isinstance(translator, cls):
            raise ValueError(f"{path_or_module} must define `translator = Translator(...)`")
        return translator

    @classmethod
    def load(cls, path) -> "Translator":
        return cls.from_module(path) if str(path).endswith(".py") else cls.from_yaml(path)
