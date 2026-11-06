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
import os
import types
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Union

import yaml

from openalea.metafspm.data_structure.configs import ScalesConfig

Factor = Union[float, int, str]
MISSING_POLICIES = ("default", "nan")         # a mean's receiving entity without contribution: its default, or NaN
MAPPING_OPTIONS = ("mask", "method")          # a link's cross mapping options (Link.mapping)

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


def _scale_name(scale: int) -> str:
    """ScalesConfig name of a scale integer."""
    from openalea.metafspm.coupling.declaration import scale_names
    names = scale_names()
    if scale not in names:
        raise ValueError(f"Scale {scale} has no ScalesConfig name")
    return names[scale]


@dataclass(frozen=True)
class Link:
    """
    ``receiver.variable`` is provided by ``provider``::

        sources:     {provider variable: factor} (weighted sum; factors may be arithmetic strings), or the provider
                     variable names passed to *formula*.
        aggregation: how values are mapped across locations ("sum", "mean", "weighted_mean", "broadcast",
                     "child", "parent", ...), with *weight* for "weighted_mean". Without it, a link between two
                     locations is mapped from its provider's kind.
        scale, source_scale:
                     the receiver-side (and sources') scale when the link changes scale (a ScalesConfig reference
                     or name); when given, the receiver's (and the sources') declared location must be that
                     scale's.
        target:      a mask of the DataStructure: the mapped values go to its entities only, the others getting the
                     receiver's default (e.g. a segment concentration broadcast to its symplastic Compartments).
        unit_check:  the declared units checked against the factors (coupling.units.check_link: the same quantity,
                     each factor the units' conversion); False for a deliberate factor (a share, a calibration).
        missing:     the value of a receiving entity no source contributes to, in a mean ("mean", "weighted_mean"):
                     "default", the receiver's default (its standalone value; e.g. an element without a liquid node
                     in a mean weighted by the liquid volume), or "nan" (NaN, to see where).
        names:       receiver-private names, {variable: private name}: the link's values are kept under the private
                     name, which the receiver reads as its variable (during its calls and read()), the providers'
                     variable of that name untouched; e.g. a gas exchange's green_area, derived from the shoot's
                     green_area and the part above the soil, names={"green_area": "exchanging_area"}.
        mapping:     a link across DataStructures (a population and a grid, the Scene's): the options of the cross
                     mapping it goes through (coupling.cross.CrossMapping), {"mask": a node mask of the plants (only
                     its entities exchange, e.g. the roots' soil nodes), "method": "barycentre" | "overlap"}; the
                     Scene infers one mapping per pair of DataStructures and options (Scene._infer_mappings).

    A source may be a geometric query (coupling.geometry.FractionIn, CentreIn): a variable the receiver's
    DataStructure derives from its elements' geometry (DataStructure.derive_geometry), e.g. the fraction of each
    element's green area above the soil.
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
    unit_check: bool = True
    missing: str = "default"
    names: Optional[Mapping[str, str]] = None
    mapping: Optional[Mapping] = None
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
        if self.missing not in MISSING_POLICIES:
            raise ValueError(f"Link {self.receiver}.{self.variable}: missing must be one of {MISSING_POLICIES}, got "
                             f"'{self.missing}'")
        object.__setattr__(self, "scale", _scale(self.scale))
        object.__setattr__(self, "source_scale", _scale(self.source_scale))
        if self.names is not None:
            names = dict(self.names)
            if set(names) != {self.variable} or not isinstance(names[self.variable], str) \
                    or names[self.variable] == self.variable:
                raise ValueError(f"Link {self.receiver}.{self.variable}: names gives its variable's private name, "
                                 f"{{'{self.variable}': 'another_name'}}, got {self.names!r}")
            object.__setattr__(self, "names", names)
        if self.mapping is not None:
            mapping = dict(self.mapping)
            unknown = set(mapping) - set(MAPPING_OPTIONS)
            if unknown:
                raise ValueError(f"Link {self.receiver}.{self.variable}: mapping options are {MAPPING_OPTIONS}, got "
                                 f"{sorted(unknown)}")
            object.__setattr__(self, "mapping", mapping)

    @property
    def private_name(self) -> str:
        """The name the link's values are kept under: its variable's private name (names=), else its variable."""
        return self.names[self.variable] if self.names else self.variable

    @property
    def mapping_options(self) -> tuple:
        """The cross mapping's options as a sorted tuple (an inferred mapping's key; () without)."""
        return tuple(sorted((self.mapping or {}).items()))
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
    def coupling(self) -> str:
        """How the link couples on one DataStructure: "shared" (the same name, nothing to do), "alias" (a rename) or
        "derived" (computed: a factor, a change of scale, a formula), as CompositeModel.couplings() lists it."""
        detail = self.mapped_detail
        return {"identity": "shared", "alias": "alias"}.get(detail, "derived")

    @property
    def mapped_detail(self) -> str:
        """
        detail of the link once its scales are checked: scale and source_scale alone only state where the
        variables are, so such a link is the identity, alias, factor, ... of its sources.
        """
        if self.detail != "scale_change" or self.aggregation is not None or self.target is not None:
            return self.detail
        return Link(self.receiver, self.variable, self.provider, self.raw_factors).detail


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

    # ── conversions ───────────────────────────────────────────────────────────

    def to_nested(self) -> dict:
        """
        Nested format {receiver: {provider: {variable: {source: factor}}}}, every pair present. A link with a
        scale, aggregation, weight, target, unit_check, missing, names or mapping is written in its long form {"sources": {...},
        "scale": name, ...}, which from_dict() reads; formulas cannot be written.
        """
        names = self.components
        nested = {receiver: {provider: {} for provider in names} for receiver in names}
        for link in self.links:
            if link.formula is not None or callable(link.aggregation) or \
                    not all(isinstance(source, str) for source in link.sources):
                raise ValueError(f"Link {link.receiver}.{link.variable} uses a formula, a callable aggregation or a "
                                 "geometric source, which the nested format cannot express")
            spec = dict(link.sources)
            options = {key: getattr(link, key) for key in ("scale", "source_scale", "aggregation", "weight", "target")
                       if getattr(link, key) is not None}
            if not link.unit_check:
                options["unit_check"] = False
            if link.missing != "default":
                options["missing"] = link.missing
            for key in ("names", "mapping"):
                if getattr(link, key) is not None:
                    options[key] = dict(getattr(link, key))
            if options:
                for key in ("scale", "source_scale"):
                    if key in options:
                        options[key] = _scale_name(options[key])
                spec = {"sources": spec, **options}
            nested[link.receiver][link.provider][link.variable] = spec
        return nested

    @classmethod
    def from_dict(cls, nested: dict) -> "Translator":
        translator = cls()
        for receiver, providers in nested.items():
            for provider, links in (providers or {}).items():
                for variable, spec in (links or {}).items():
                    if isinstance(spec, Mapping) and "sources" in spec:
                        options = {key: spec[key] for key in ("scale", "source_scale", "aggregation", "weight", "target",
                                                              "unit_check", "missing", "names", "mapping")
                                   if key in spec}
                        translator.link(receiver, variable, provider, dict(spec["sources"]), **options)
                    else:
                        translator.link(receiver, variable, provider, dict(spec))
        return translator

    @classmethod
    def load(cls, source=None) -> "Translator":
        """
        A Translator from *source*: a Translator (as is), a nested dict (from_dict), a Python module or the path of
        one (.py) defining ``translator = Translator(...)``, or the path of a YAML file of the nested format. None:
        no link.
        """
        if source is None:
            return cls()
        if isinstance(source, cls):
            return source
        if isinstance(source, Mapping):
            return cls.from_dict(source)
        if isinstance(source, types.ModuleType):
            return cls._from_module(source)
        path = os.fspath(source)
        if os.path.isdir(path):                          # Windows' open() raises PermissionError on a directory
            raise IsADirectoryError(f"the translator path is a directory: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no translator at '{path}' (a .py module or a YAML file; None for no link)")
        return cls._from_module(path) if path.endswith(".py") else cls._from_yaml(path)

    @classmethod
    def _from_yaml(cls, path) -> "Translator":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def _from_module(cls, path_or_module) -> "Translator":
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
