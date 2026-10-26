"""
Units of the declarations, read as dimensions and a scale, for the checks of the translator links.

The grammar (declarations write it so): factors separated by spaces, each a symbol with an optional signed integer
exponent (`mol m-2 s-1`, `mm3 s-1 MPa-1`), dimensionless written `adim`; what the amount is of goes in
`unit_comment` (`unit="mol s-1", unit_comment="of hexose"`). Read too, for the older spellings: dots or `·` between
factors, `^` or `**` before exponents, `-`, `1` and `dimensionless` for adim, "of <name>" qualifiers (ignored),
"per (...)" (the group's exponents negated), `%`.

    parse_unit("mmol m-2 s-1")            ((−2, 0, −1, 1, 0), 1e-3): dimensions (m, kg, s, mol, K) and SI scale
    conversion_factor("cm", "m")          100.0: a value in m times it is the value in cm
"""
import math
import re
from typing import Optional

# dimensions: length, mass, time, amount, temperature
_ONE = (0, 0, 0, 0, 0)
_UNITS = {
    "m": ((1, 0, 0, 0, 0), 1.), "g": ((0, 1, 0, 0, 0), 1e-3), "s": ((0, 0, 1, 0, 0), 1.),
    "mol": ((0, 0, 0, 1, 0), 1.), "K": ((0, 0, 0, 0, 1), 1.), "°C": ((0, 0, 0, 0, 1), 1.),
    "min": ((0, 0, 1, 0, 0), 60.), "h": ((0, 0, 1, 0, 0), 3600.), "d": ((0, 0, 1, 0, 0), 86400.),
    "day": ((0, 0, 1, 0, 0), 86400.), "yr": ((0, 0, 1, 0, 0), 365.25 * 86400.),
    "L": ((3, 0, 0, 0, 0), 1e-3), "l": ((3, 0, 0, 0, 0), 1e-3),
    "Pa": ((-1, 1, -2, 0, 0), 1.), "bar": ((-1, 1, -2, 0, 0), 1e5), "N": ((1, 1, -2, 0, 0), 1.),
    "J": ((2, 1, -2, 0, 0), 1.), "W": ((2, 1, -3, 0, 0), 1.),
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
    if len(text) > 1 and text[0] in _PREFIXES and text[1:] in _UNITS and text[1:] not in ("°C", "°", "%", "adim"):
        dimensions, scale = _UNITS[text[1:]]
        return dimensions, scale * _PREFIXES[text[0]]
    return None


def parse_unit(unit) -> Optional[tuple]:
    """((length, mass, time, amount, temperature) exponents, SI scale) of a declared unit; None when undeclared
    ("" or None) or not readable."""
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
    dimensions, scale = [0, 0, 0, 0, 0], 1.
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
        for axis in range(5):
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


def check_link(received_unit, provided_unit, factor, name: str) -> None:
    """A translator link's units against its coefficient: undeclared units agree with anything; readable units must
    be the same quantity and the coefficient (its absolute value) their conversion factor; unreadable ones must be
    spelled alike when the coefficient is ±1. Raises ValueError naming the link."""
    received, provided = parse_unit(received_unit), parse_unit(provided_unit)
    if received is not None and provided is not None:
        if received[0] != provided[0]:
            raise ValueError(f"{name}: '{provided_unit}' cannot feed '{received_unit}', not the same quantity")
        expected = provided[1] / received[1]
        if not math.isclose(abs(float(factor)), expected, rel_tol=1e-9):
            raise ValueError(f"{name}: the coefficient {factor} does not convert '{provided_unit}' into "
                             f"'{received_unit}' ({expected:g}); give it, or unit_check=False for a deliberate one")
        return
    from openalea.metafspm.coupling.declaration import units_agree
    if factor in (1., -1.) and not units_agree(received_unit, provided_unit):
        raise ValueError(f"{name}: the units differ ('{received_unit}' <- '{provided_unit}') and the link has no "
                         "conversion factor")
