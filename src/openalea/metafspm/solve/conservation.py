"""
The quantities a graph system conserves, declared with it and checked at every step (survey 13)::

  @graph_system(node_unknowns=["pressure_head"], integrate="adaptive",
                conserved={"water": Conservation(amount=("water_content", "volume"), inflow=["surface_inflow"],
                                                  outflow=["drainage", "plant_uptake"], unit="m3")})
  class _water: ...

  component.conservation()        {(system, quantity): {"inflow", "outflow", "change", "residual"}} since the last
                                  reset (SceneRecorder writes it after every step: conservation.csv)

Over each call of the system the framework integrates the exchanged amounts, the inflow and outflow rates (per
second) at the end of each accepted sub-step times its length (an implicit step's: what its balance holds, so that
the residual is the solver's tolerance), over the solved entities (each connected piece's own with split=
"components"), and measures the amount held before and after: residual = change - (inflow - outflow). A rejected
sub-step of an adaptive integration counts for nothing.

A term is a variable name, or a tuple of names (their product per entity: θ × V), at the system's nodes or at a coarse
location (each entity counted for the solved nodes it owns); amount None: nothing held (a steady system: the
residual is outflow - inflow). The amounts are in *unit* (default: the amount's declared unit, its factors'
multiplied; without amount, the first inflow's times s), each rate's declared unit times s converted to it.
"""
import numpy as np

from openalea.metafspm.coupling.units import conversion_factor, parse_unit


class Conservation:
    """A conserved quantity of a graph system (see the module): *amount*, *inflow*, *outflow*, *unit*."""

    def __init__(self, amount=None, inflow=(), outflow=(), unit: str = None):
        self.amount = None if amount is None else _term(amount, "amount")
        self.inflow = tuple(_term(term, "inflow") for term in _terms(inflow))
        self.outflow = tuple(_term(term, "outflow") for term in _terms(outflow))
        if self.amount is None and not self.inflow and not self.outflow:
            raise ValueError("Conservation: give what is held (amount=) or what crosses (inflow=, outflow=)")
        self.unit = unit

    def __repr__(self) -> str:
        return f"Conservation(amount={self.amount}, inflow={self.inflow}, outflow={self.outflow}, unit={self.unit!r})"

    def names(self) -> set:
        terms = ((self.amount,) if self.amount is not None else ()) + self.inflow + self.outflow
        return {name for term in terms for name in term}


def _terms(value) -> tuple:
    return (value,) if isinstance(value, str) else tuple(value)


def _term(value, role: str) -> tuple:
    names = (value,) if isinstance(value, str) else tuple(value)
    if not names or not all(isinstance(name, str) for name in names):
        raise TypeError(f"Conservation({role}=...): a variable name or a tuple of names (their product), got "
                        f"{value!r}")
    return names


def check_conserved(conserved) -> dict:
    """graph_system's conserved= as {quantity: Conservation}."""
    if conserved is None:
        return {}
    if not isinstance(conserved, dict) or not all(isinstance(spec, Conservation) for spec in conserved.values()):
        raise TypeError("graph_system(conserved=...): {quantity: Conservation(amount=, inflow=, outflow=, unit=)}")
    return dict(conserved)


# ── during a solve ──

def _declared_unit(component, name):
    """The unit the component declares for *name* (its field's), None when it does not declare it."""
    from dataclasses import fields, is_dataclass
    if not is_dataclass(component):
        return None
    return next((f.metadata.get("unit") for f in fields(component) if f.name == name), None)


def _term_unit(component, term) -> str:
    units = [_declared_unit(component, name) for name in term]
    if any(unit is None or parse_unit(unit) is None for unit in units):
        return None
    return " ".join(units)


def _units(component, quantity: str, spec: Conservation) -> dict:
    """{term: factor to the quantity's unit} (rates: their unit times s), and the unit; once per component."""
    cache = component.caches().setdefault("conservation_units", {})
    if quantity in cache:
        return cache[quantity]
    unit = spec.unit
    if unit is None and spec.amount is not None:
        unit = _term_unit(component, spec.amount)
    elif unit is None:
        first = _term_unit(component, (spec.inflow + spec.outflow)[0])
        unit = None if first is None else f"{first} s"
    factors = {}
    for term, rate in ([(spec.amount, False)] if spec.amount is not None else []) + \
            [(term, True) for term in spec.inflow + spec.outflow]:
        own = _term_unit(component, term)
        own = None if own is None else (f"{own} s" if rate else own)
        try:
            factor = conversion_factor(unit, own)
        except ValueError as error:
            raise ValueError(f"{type(component).__name__}: conserved '{quantity}': {' x '.join(term)} "
                             f"({own}) is no amount of {unit}: {error}") from None
        factors[term] = 1. if factor is None else factor
    cache[quantity] = (factors, unit)
    return cache[quantity]


def _values(component, data_structure, term, restriction) -> np.ndarray:
    """The term's values (the product of its variables) on the solved entities: the restriction's nodes, or the
    entities at a coarse location owning them; all of them without a restriction."""
    product = None
    location = None
    for name in term:
        values = np.asarray(data_structure.get(name), dtype=float)
        here = data_structure.location(name)
        if location is None:
            location = here
        elif here != location:
            raise ValueError(f"{type(component).__name__}: a conserved term's variables {term} are at "
                             f"different locations ({location}, {here})")
        values = values.reshape(-1) if here != "scalar" else values.reshape(1)
        product = values if product is None else product * values
    if restriction is None or location == "scalar":
        return product
    if location in ("node", "cell"):
        return product[restriction.node_idx]
    if location == "edge":
        raise ValueError(f"{type(component).__name__}: a conserved term at the edges ({term}): give node terms")
    owner = np.asarray(data_structure.owner(location)).reshape(-1)[restriction.node_idx]
    return product[np.unique(owner[owner >= 0])]


def held(component, data_structure, conserved) -> dict:
    """{quantity: the amount held} on every entity (in the quantity's unit; 0 without amount)."""
    out = {}
    for quantity, spec in conserved.items():
        if spec.amount is None:
            out[quantity] = 0.
            continue
        factors, _ = _units(component, quantity, spec)
        out[quantity] = float(np.sum(_values(component, data_structure, spec.amount, None))) * factors[spec.amount]
    return out


def record_substep(component, data_structure, conserved, h: float) -> None:
    """A solved (sub-)step of length *h*: its inflow and outflow (their rates now, on the solved entities, times
    h), pending until the integration accepts it (accept, reject)."""
    restriction = component.__dict__.get("_restriction")
    flows = {}
    for quantity, spec in conserved.items():
        factors, _ = _units(component, quantity, spec)
        flows[quantity] = (sum(float(np.sum(_values(component, data_structure, term, restriction))) * factors[term]
                               for term in spec.inflow) * h,
                           sum(float(np.sum(_values(component, data_structure, term, restriction))) * factors[term]
                               for term in spec.outflow) * h)
    component.__dict__.setdefault("_conservation_pending", []).append(flows)


def _mark(component) -> int:
    """The number of pending sub-steps (an adaptive trial's start)."""
    return len(component.__dict__.get("_conservation_pending", []))


def _reject(component, since: int) -> None:
    """The pending sub-steps from *since* on dropped (a rejected or replaced trial)."""
    pending = component.__dict__.get("_conservation_pending")
    if pending is not None:
        del pending[since:]


def _begin_call(component, data_structure, method_name: str, conserved) -> None:
    """Before a call of the system: the amounts held, the pending sub-steps emptied."""
    component.__dict__["_conservation_pending"] = []
    component.__dict__.setdefault("_conservation_start", {}).setdefault(method_name, held(component, data_structure, conserved))


def _end_call(component, data_structure, method_name: str, conserved) -> None:
    """After a call: its accepted sub-steps' flows added to the balance since the last reset."""
    pending = component.__dict__.pop("_conservation_pending", [])
    end = held(component, data_structure, conserved)
    balances = component.__dict__.setdefault("_conservation", {})
    start = component.__dict__["_conservation_start"][method_name]
    for quantity, spec in conserved.items():
        inflow = sum(flows[quantity][0] for flows in pending)
        outflow = sum(flows[quantity][1] for flows in pending)
        entry = balances.setdefault((method_name, quantity), {"inflow": 0., "outflow": 0., "change": 0.,
                                                              "residual": 0., "unit": _units(component, quantity,
                                                                                             spec)[1]})
        entry["inflow"] += inflow
        entry["outflow"] += outflow
        entry["change"] = end[quantity] - start[quantity]
        entry["residual"] = entry["change"] - (entry["inflow"] - entry["outflow"])


def balances(component, reset: bool = False) -> dict:
    """{(system, quantity): {"inflow", "outflow", "change", "residual", "unit"}} since the last reset (*reset*: start a
    new balance after reading this one)."""
    out = {key: dict(entry) for key, entry in component.__dict__.get("_conservation", {}).items()}
    if reset:
        component.__dict__.pop("_conservation", None)
        component.__dict__.pop("_conservation_start", None)
    return out
