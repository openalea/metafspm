"""
Units of the declarations, read as dimensions and a scale, for the checks of the translator links.

The grammar (declarations write it so): factors separated by spaces, each a symbol with an optional signed integer
exponent (`mol m-2 s-1`, `mm3 s-1 MPa-1`), dimensionless written `adim`; what the amount is of goes in
`unit_comment` (`unit="mol s-1", unit_comment="of hexose"`). Read too, for the older spellings: dots or `·` between
factors, `^` or `**` before exponents, `-`, `1` and `dimensionless` for adim, "of <name>" qualifiers (ignored),
"per (...)" (the group's exponents negated), `%`.

Thermal time is a temperature times a time: `°Cd` is `°C d` (degree-days, a temperature difference: no offset). A
duration counted in phyllochrons (`phyllochron`, the thermal time between two successive leaves) has its own
dimension: its number of degree-days is the plant's phyllochron, a parameter, not a constant, so a link between
phyllochrons and degree-days is refused.

    parse_unit("mmol m-2 s-1")            ((−2, 0, −1, 1, 0, 0), 1e-3): dimensions (m, kg, s, mol, K, phyllochron)
                                          and SI scale
    conversion_factor("cm", "m")          100.0: a value in m times it is the value in cm

A declaration's unit is checked when its component class is defined (Component.__init_subclass__): a unit this
grammar cannot read raises, naming the component, the variable and the unit. A link is checked when the models are
built: both units must be declared and readable (unit_check=False on a link skips it).

A link between two quantities that are not the same (an amount of water and its volume, a diameter and a radius) is
unit-checked through a named equivalence, the physical factor between them with its unit (Equivalence: the water's
molar volume, 18e-3 cm3 mmol-1): the link's factor is the equivalence times the conversion of the provider's unit,
times the equivalence's, into the receiver's, computed and checked by metafspm (a link {"transpiration":
WATER_MOLAR_VOLUME} from mmol s-1 to cm3 d-1 gets 18e-3 × 86400).
"""
import math
import re
from dataclasses import dataclass
from typing import Optional

# dimensions: length, mass, time, amount, temperature, phyllochron (a duration counted in phyllochrons)
_ONE = (0, 0, 0, 0, 0, 0)
_UNITS = {
    "m": ((1, 0, 0, 0, 0, 0), 1.), "g": ((0, 1, 0, 0, 0, 0), 1e-3), "s": ((0, 0, 1, 0, 0, 0), 1.),
    "mol": ((0, 0, 0, 1, 0, 0), 1.), "K": ((0, 0, 0, 0, 1, 0), 1.), "°C": ((0, 0, 0, 0, 1, 0), 1.),
    "°Cd": ((0, 0, 1, 0, 1, 0), 86400.), "phyllochron": ((0, 0, 0, 0, 0, 1), 1.),
    "min": ((0, 0, 1, 0, 0, 0), 60.), "h": ((0, 0, 1, 0, 0, 0), 3600.), "d": ((0, 0, 1, 0, 0, 0), 86400.),
    "day": ((0, 0, 1, 0, 0, 0), 86400.), "yr": ((0, 0, 1, 0, 0, 0), 365.25 * 86400.),
    "L": ((3, 0, 0, 0, 0, 0), 1e-3), "l": ((3, 0, 0, 0, 0, 0), 1e-3),
    "Pa": ((-1, 1, -2, 0, 0, 0), 1.), "bar": ((-1, 1, -2, 0, 0, 0), 1e5), "N": ((1, 1, -2, 0, 0, 0), 1.),
    "J": ((2, 1, -2, 0, 0, 0), 1.), "W": ((2, 1, -3, 0, 0, 0), 1.),
    "°": (_ONE, math.pi / 180.), "rad": (_ONE, 1.), "adim": (_ONE, 1.), "%": (_ONE, 0.01),
    "pixel": (_ONE, 1.),
}
_PREFIXES = {"k": 1e3, "M": 1e6, "G": 1e9, "h": 1e2, "c": 1e-2, "m": 1e-3, "µ": 1e-6, "u": 1e-6, "n": 1e-9,
             "p": 1e-12}
_DIMENSIONLESS = {"adim", "-", "1", "dimensionless", "adimensional", "unitless"}
_FACTOR = re.compile(r"^(.+?)(-?\d+)?$")
_OF_ELEMENT = re.compile(r"^([kmµu]?(?:g|mol))([A-Z][A-Za-z0-9]*)$")      # "gC", "molN": a unit of an element


def _symbol(text: str):
    """(dimensions, scale) of a unit symbol, possibly prefixed, or None."""
    if text in _UNITS:
        return _UNITS[text]
    element = _OF_ELEMENT.match(text)
    if element:
        return _symbol(element.group(1))
    if len(text) > 1 and text[0] in _PREFIXES and text[1:] in _UNITS and text[1:] not in ("°C", "°Cd", "°", "%", "adim"):
        dimensions, scale = _UNITS[text[1:]]
        return dimensions, scale * _PREFIXES[text[0]]
    return None


def parse_unit(unit) -> Optional[tuple]:
    """((length, mass, time, amount, temperature, phyllochron) exponents, SI scale) of a declared unit; None when
    undeclared ("" or None) or not readable."""
    if unit is None:
        return None
    text = str(unit).strip()
    if not text:
        return None
    if text in _DIMENSIONLESS:
        return _ONE, 1.
    text = text.replace("**", "").replace("^", "").replace("·", " ").replace("..", " ").replace("/", " per ")
    text = text.replace("(", " ( ").replace(")", " ) ")
    text = re.sub(r"(?<=[A-Za-zµ°%\d])\.(?=[A-Za-zµ°%])", " ", text)          # "mol.m-2" → "mol m-2"
    tokens = [token for token in text.split() if token not in (".", "..")]
    dimensions, scale = [0] * len(_ONE), 1.
    sign, in_group, group_sign = 1, False, 1
    skip_next = False
    for token in tokens:
        if skip_next:                                   # "of <name>": a qualifier
            skip_next = False
            continue
        if token == "of":
            skip_next = True
            continue
        if token == "per":
            sign = -1
            continue
        if token == "(":
            in_group, group_sign = True, sign
            continue
        if token == ")":
            in_group, sign = False, 1
            continue
        token = token.strip(".")
        if not token:
            continue
        match = _FACTOR.match(token)
        name, exponent = match.group(1), int(match.group(2)) if match.group(2) else 1
        found = _symbol(name)
        if found is None:
            return None
        exponent *= group_sign if in_group else sign
        if not in_group and sign == -1:
            sign = 1                                    # "per x": only the next factor
        for axis in range(len(_ONE)):
            dimensions[axis] += found[0][axis] * exponent
        scale *= found[1] ** exponent
    return tuple(dimensions), scale


def conversion_factor(to_unit, from_unit) -> Optional[float]:
    """The factor turning a value in *from_unit* into *to_unit* (a link's coefficient); None when either is
    undeclared or not readable. Raises ValueError when their dimensions differ."""
    to, source = parse_unit(to_unit), parse_unit(from_unit)
    if to is None or source is None:
        return None
    if to[0] != source[0]:
        raise ValueError(f"'{from_unit}' and '{to_unit}' are not the same quantity")
    return source[1] / to[1]


def canonical(unit) -> Optional[str]:
    """A readable unit in the declarations' grammar (spaces between factors, `adim`), or None when not readable;
    "of <name>" qualifiers dropped (they go to unit_comment)."""
    if unit is None or not str(unit).strip():
        return unit
    text = str(unit).strip()
    if text in _DIMENSIONLESS:
        return "adim"
    if parse_unit(text) is None:
        return None
    text = text.replace("**", "").replace("^", "").replace("·", " ").replace("..", " ")
    text = re.sub(r"(?<=[A-Za-zµ°%\d])\.(?=[A-Za-zµ°%])", " ", text)
    text = re.sub(r"\s+of\s+[^\s()]+", "", text)
    return " ".join(token.strip(".") for token in text.split() if token.strip("."))


def check_readable(received_unit, provided_unit, name: str) -> tuple:
    """The parsed units of a link's receiver and provider (parse_unit), a ValueError naming the link when either is
    undeclared or unreadable: such units cannot be compared."""
    received, provided = parse_unit(received_unit), parse_unit(provided_unit)
    for unit, parsed, side in ((received_unit, received, "receiver"), (provided_unit, provided, "provider")):
        if parsed is None:
            state = ("has no declared unit" if unit is None or not str(unit).strip()
                     else f"has the unit '{unit}', which is not readable")
            raise ValueError(f"{name}: the {side} {state}, so the link cannot be unit-checked; declare its unit "
                             "(\"adim\" for a dimensionless one), or unit_check=False on a deliberate link")
    return received, provided


def check_link_units(received_unit, provided_unit, factor, name: str) -> None:
    """A translator link's units against its coefficient: both declared and readable (check_readable), the same
    quantity, and the coefficient (its absolute value) their conversion factor. Raises ValueError naming the link."""
    received, provided = check_readable(received_unit, provided_unit, name)
    if received[0] != provided[0]:
        raise ValueError(f"{name}: '{provided_unit}' cannot feed '{received_unit}', not the same quantity")
    expected = provided[1] / received[1]
    if not math.isclose(abs(float(factor)), expected, rel_tol=1e-9):
        raise ValueError(f"{name}: the coefficient {factor} does not convert '{provided_unit}' into "
                         f"'{received_unit}' ({expected:g}); give it, or unit_check=False for a deliberate one")


def check_declared_unit(unit, name: str) -> None:
    """A declaration's unit: empty (undeclared) or readable by the grammar; a ValueError naming *name* (its
    "<Component>.<variable>") and the unit otherwise."""
    if unit is None or not str(unit).strip() or parse_unit(unit) is not None:
        return
    raise ValueError(f"{name}: unit '{unit}' is not readable by metafspm's unit grammar (coupling.units: factors "
                     "separated by spaces with integer exponents, 'mol m-2 s-1'; 'adim' for dimensionless; what an "
                     "amount is of in unit_comment)")


@dataclass(frozen=True)
class Equivalence:
    """
    A named physical factor between two quantities (*value* in *unit*: what a link's provider unit times it is the
    receiver's quantity), a link's factor checked by its units (Equivalence.factor): the water's molar volume
    turning an amount of water into its volume, a radius from its diameter. *references*: its source.
    """
    name: str
    value: float
    unit: str = "adim"
    references: str = ""

    def __str__(self) -> str:
        return self.name

    def factor(self, received_unit, provided_unit, link: str = "") -> float:
        """The link's factor from *provided_unit* to *received_unit* through the equivalence: its value times the
        conversion of the provided unit times its own into the received one. ValueError when a unit is undeclared or
        unreadable, or when the quantities do not match."""
        received, provided, own = parse_unit(received_unit), parse_unit(provided_unit), parse_unit(self.unit)
        if received is None or provided is None or own is None:
            raise ValueError(f"{link}: the equivalence '{self.name}' converts declared units (got '{provided_unit}' "
                             f"into '{received_unit}', the equivalence in '{self.unit}')")
        product = tuple(a + b for a, b in zip(provided[0], own[0]))
        if product != received[0]:
            raise ValueError(f"{link}: '{provided_unit}' times the equivalence '{self.name}' ({self.unit}) is not "
                             f"'{received_unit}'")
        return float(self.value) * (provided[1] * own[1] / received[1])


WATER_MOLAR_VOLUME = Equivalence("water_molar_volume", 18e-3, "cm3 mmol-1",
                                 references="18 g mol-1, 1 g of liquid water 1 cm3")
RADIUS_OF_DIAMETER = Equivalence("radius_of_diameter", 0.5, "adim", references="geometry: a radius is half the "
                                                                               "diameter")

# The named equivalences a link's factor may give by name ({"diameter": "radius_of_diameter"}, a YAML translator's)
EQUIVALENCES = {equivalence.name: equivalence for equivalence in (WATER_MOLAR_VOLUME, RADIUS_OF_DIAMETER)}


def register_equivalence(equivalence: Equivalence) -> Equivalence:
    """Name *equivalence* for the links (EQUIVALENCES); it is returned."""
    if not isinstance(equivalence, Equivalence):
        raise TypeError(f"register_equivalence: an Equivalence, got {equivalence!r}")
    known = EQUIVALENCES.get(equivalence.name)
    if known is not None and known != equivalence:
        raise ValueError(f"an equivalence '{equivalence.name}' is already registered ({known})")
    EQUIVALENCES[equivalence.name] = equivalence
    return equivalence
