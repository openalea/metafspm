"""
Forcings, the time-varying inputs of a scene (meteo, a measured temperature, a shoot's supply)::

  Scene(..., forcings={"PARi": series, "soil_temperature": 15., "struct_mass_input": function})
  Scene(..., forcings=table)                  a DataFrame indexed by time (s), one column per forcing

A forcing is a number; a callable of the scene time (s); a pandas Series indexed by time (s); or a (times, values)
pair. Tables are linearly interpolated (and held constant beyond their ends). A callable of the time and the entity
ids, (time, ids) -> one value per entity, varies in space (a soil temperature by depth, an input per segment): the
Scene writes it into the input or parameter of that name, on the entities of its location. A component declaring
the name with forcing_variable reads it with self.forcing(name); otherwise the Scene writes it, at the start of each step, into
the input or parameter of that name of every component declaring it (not provided by a coupling): its mean over the
step.

A forcing is read over a time interval, not at one instant, so that the results do not depend on the time step: a
component reads its mean over the (sub-)step it computes (forcing_variable(aggregate="mean"), the default), its total
over it (aggregate="total": a rain's depth from its rate), or its value at the step's end (aggregate="instant"). The
mean is exact for numbers, tables (piecewise linear) and pulses; for other callables the 4-point Gauss-Legendre rule on
panels of at most an hour (exact for polynomials of degree 7 on each: a daily course read hour by hour or day by day;
a forcing with steps, a rain shower within a day, is a pulse, whose mean is exact at any step)::

  as_forcing(value, time_unit=86400.)         a user's input in days (a number, a function of the day, a Series
                                              indexed by day, one value per day from day 0) as a forcing
  pulse(start, end, total)                    a rate forcing from a total (a rain of 5 mm from 14 to 15 h): total /
                                              (end - start) on (start, end], 0 elsewhere
  forcing_value(forcing, time)                its value at *time* (s)
  forcing_mean(forcing, start, end)           its mean over [start, end] (s); forcing_total its integral
"""
import inspect
from typing import Mapping

import numpy as np


def _series(value) -> bool:
    return hasattr(value, "index") and hasattr(value, "to_numpy")


def is_spatial(forcing) -> bool:
    """Whether *forcing* varies in space: a callable of two required arguments, (time, ids)."""
    if not callable(forcing) or isinstance(forcing, _InUnit):
        return False
    try:
        parameters = inspect.signature(forcing).parameters.values()
    except (TypeError, ValueError):
        return False
    required = [p for p in parameters if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY,
                                                                             p.POSITIONAL_OR_KEYWORD)]
    return len(required) >= 2


def forcing_value(forcing, time: float) -> float:
    """The value of *forcing* at *time* (s)."""
    if callable(forcing):
        return forcing(time)
    if np.ndim(forcing) == 0 and not _series(forcing):
        return float(forcing)
    if _series(forcing):                                         # a Series indexed by time
        times, values = np.asarray(forcing.index, dtype=np.float64), np.asarray(forcing, dtype=np.float64)
    else:
        times, values = (np.asarray(x, dtype=np.float64) for x in forcing)
    return float(np.interp(time, times, values))


# The 4-point Gauss-Legendre rule on [-1, 1]
_GAUSS_NODES = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
_GAUSS_WEIGHTS = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])


def gauss_points(start: float, end: float) -> tuple:
    """(times, weights) of the 4-point Gauss-Legendre rule on [*start*, *end*], the weights summing to 1 (a mean)."""
    middle, half = 0.5 * (start + end), 0.5 * (end - start)
    return middle + half * _GAUSS_NODES, 0.5 * _GAUSS_WEIGHTS


QUADRATURE_PANEL = 3600.        # s: the longest interval one 4-point Gauss-Legendre rule averages a function over


def panels(start: float, end: float) -> np.ndarray:
    """The edges of the panels of at most QUADRATURE_PANEL covering [*start*, *end*] (equal panels: one for an hourly
    step, 24 for a daily one); [start, end] for an empty interval."""
    count = max(1, int(np.ceil((end - start) / QUADRATURE_PANEL - 1e-9)))
    return np.linspace(start, end, count + 1)


def quadrature(start: float, end: float) -> tuple:
    """(times, weights) of the mean over [*start*, *end*]: the 4-point Gauss-Legendre rule on panels of at most
    QUADRATURE_PANEL (an hour: one panel for an hourly step, 24 for a daily one, so that a daily course is averaged
    as well at any step), the weights summing to 1."""
    edges = panels(start, end)
    points = [gauss_points(first, last) for first, last in zip(edges[:-1], edges[1:])]
    return (np.concatenate([times for times, _ in points]),
            np.concatenate([weights for _, weights in points]) / (edges.size - 1))


def _gauss_mean(function, start: float, end: float) -> float:
    """The mean of *function* (of the time) over [*start*, *end*] (quadrature()); its value when the points agree (a
    constant stays exact)."""
    times, weights = quadrature(start, end)
    values = [float(function(time)) for time in times]
    if all(value == values[0] for value in values):
        return values[0]
    return float(np.dot(weights, values))


def spatial_mean(forcing, ids, start: float, end: float) -> np.ndarray:
    """The mean over [*start*, *end*] of a forcing varying in space, (time, ids) -> one value per entity, for each of
    *ids* (quadrature(); the values themselves where they do not change over it)."""
    if not end > start:
        return np.asarray(forcing(end, ids), dtype=float)
    times, weights = quadrature(start, end)
    values = [np.asarray(forcing(time, ids), dtype=float) for time in times]
    if all(np.array_equal(value, values[0]) for value in values[1:]):
        return values[0]
    return sum(weight * value for weight, value in zip(weights, values))


def _table_mean(times, values, start: float, end: float) -> float:
    """The exact mean over [*start*, *end*] of the table (*times*, *values*) linearly interpolated, held constant beyond
    its ends."""
    inside = times[(times > start) & (times < end)]
    points = np.concatenate([[start], inside, [end]])
    at = np.interp(points, times, values)
    return float(np.sum(0.5 * (at[1:] + at[:-1]) * np.diff(points)) / (end - start))


def _table(forcing) -> tuple:
    """(times, values) of a table forcing: a Series indexed by time, or a (times, values) pair."""
    if _series(forcing):
        return np.asarray(forcing.index, dtype=np.float64), np.asarray(forcing, dtype=np.float64)
    return tuple(np.asarray(x, dtype=np.float64) for x in forcing)


def forcing_mean(forcing, start: float, end: float) -> float:
    """
    The mean of *forcing* over [*start*, *end*] (s): exact for a number, a table (piecewise linear, held constant
    beyond its ends) and a forcing with its own mean(start, end) (pulse, as_forcing's); for any other callable, the
    4-point Gauss-Legendre rule on panels of at most an hour (QUADRATURE_PANEL). Its value at *end* when the interval
    is empty.
    """
    start, end = float(start), float(end)
    if not end > start:
        return float(forcing_value(forcing, end))
    if callable(getattr(forcing, "mean", None)) and not _series(forcing):
        return float(forcing.mean(start, end))
    if callable(forcing):
        return _gauss_mean(forcing, start, end)
    if np.ndim(forcing) == 0 and not _series(forcing):
        return float(forcing)
    return _table_mean(*_table(forcing), start, end)


def forcing_total(forcing, start: float, end: float) -> float:
    """The integral of *forcing* (a rate) over [*start*, *end*] (s): its mean times the duration."""
    return forcing_mean(forcing, start, end) * max(0., float(end) - float(start))


AGGREGATES = ("mean", "total", "instant")


def forcing_over(forcing, start: float, end: float, aggregate: str = "mean") -> float:
    """*forcing* over [*start*, *end*] (s) by *aggregate*: "mean", "total" (the integral of a rate) or "instant" (its
    value at *end*)."""
    if aggregate == "mean":
        return forcing_mean(forcing, start, end)
    if aggregate == "total":
        return forcing_total(forcing, start, end)
    if aggregate == "instant":
        return float(forcing_value(forcing, end))
    raise ValueError(f"aggregate '{aggregate}': one of {AGGREGATES}")


class pulse:
    """
    A rate forcing given by its *total* over (*start*, *end*] (s of scene time): total / (end - start) there, 0
    elsewhere (a rain of 5 mm from 14 to 15 h: pulse(14 * 3600, 15 * 3600, 5e-3), in m s-1). Its mean over any
    interval is exact (the share of the pulse inside it), so a step longer than the pulse gets its whole total.
    Picklable (checkpoints).
    """

    def __init__(self, start: float, end: float, total: float):
        self.start, self.end, self.total = float(start), float(end), float(total)
        if not self.end > self.start:
            raise ValueError(f"pulse({start}, {end}, {total}): its end after its start")

    @property
    def rate(self) -> float:
        return self.total / (self.end - self.start)

    def __call__(self, time: float) -> float:
        return self.rate if self.start < float(time) <= self.end else 0.

    def mean(self, start: float, end: float) -> float:
        start, end = float(start), float(end)
        if not end > start:
            return self(end)
        overlap = max(0., min(end, self.end) - max(start, self.start))
        return self.rate * overlap / (end - start)

    def __repr__(self) -> str:
        return f"pulse({self.start:g}, {self.end:g}, {self.total:g})"


class _InUnit:
    """A user's input in their time unit, read at the scene time (picklable, for checkpoints)."""

    def __init__(self, value, time_unit: float, origin):
        self.value, self.time_unit = value, float(time_unit)
        self.origin = origin if origin == "calendar" else float(origin)

    def at_calendar(self, start) -> None:
        """Resolve origin "calendar" from the Scene's start (day of year, hour): the day of year for an input in days,
        the time of day otherwise."""
        if self.origin == "calendar":
            day, hour = start
            self.origin = (float(day) + float(hour) / 24.) if self.time_unit >= 86400. else \
                float(hour) * 3600. / self.time_unit

    def _when(self, time: float) -> float:
        if self.origin == "calendar":
            raise ValueError("a forcing of origin 'calendar' needs the Scene's start=(day_of_year, hour)")
        return self.origin + float(time) / self.time_unit

    def mean(self, start: float, end: float) -> float:
        """The mean over [*start*, *end*] (s of scene time), in the input's unit: exact for a table, a sequence and a
        value with its own mean, a Gauss-Legendre quadrature for a function."""
        first, last = self._when(start), self._when(end)
        value = self.value
        if callable(getattr(value, "mean", None)) and not _series(value) and not isinstance(value, np.ndarray):
            return float(value.mean(first, last))
        if _series(value):
            return _table_mean(np.asarray(value.index, dtype=float), np.asarray(value, dtype=float), first, last)
        if np.ndim(value) > 0 and not callable(value):
            values = np.asarray(value, dtype=float)
            return _table_mean(np.arange(values.size, dtype=float), values, first, last)
        if callable(value):
            return _gauss_mean(self, start, end)
        return float(value)

    def __call__(self, time: float) -> float:
        when = self._when(time)
        value = self.value
        if callable(value):
            return float(value(when))
        if _series(value):
            return float(np.interp(when, np.asarray(value.index, dtype=float), np.asarray(value, dtype=float)))
        if np.ndim(value) > 0:
            values = np.asarray(value, dtype=float)
            return float(np.interp(when, np.arange(values.size), values))
        return float(value)


TIME_UNITS = {"s": 1., "min": 60., "h": 3600., "d": 86400.}


def seconds(time_unit) -> float:
    """A time unit in seconds: a number of seconds, or its label ("s", "min", "h", "d")."""
    if isinstance(time_unit, str):
        if time_unit not in TIME_UNITS:
            raise ValueError(f"time unit '{time_unit}': one of {sorted(TIME_UNITS)}, or a number of seconds")
        return TIME_UNITS[time_unit]
    return float(time_unit)


def time_unit_label(time_unit) -> str:
    """A time unit's label ("s", "min", "h", "d"), from its label or its number of seconds."""
    if isinstance(time_unit, str):
        seconds(time_unit)
        return time_unit
    for label, value in TIME_UNITS.items():
        if float(time_unit) == value:
            return label
    raise ValueError(f"time unit {time_unit} s has no label: one of {TIME_UNITS}")


def as_forcing(value, time_unit=1., origin: float = 0.):
    """
    A forcing of the scene time from a user's input given in *time_unit* (s, or "s", "min", "h", "d"; 86400 or "d"
    for days): a number, a function
    of that unit, a pandas Series indexed by it, or a sequence (one value per unit from 0), linearly interpolated.
    *origin*: the input's time at the scene's time 0 (e.g. the hour of the day the scene starts), or "calendar": the
    Scene's start (Scene(start=(day_of_year, hour))), its time of day for an input by the hour (a daily course),
    its day of year for an input by the day.
    """
    if np.ndim(value) == 0 and not callable(value) and not _series(value):
        return float(value)
    return _InUnit(value, seconds(time_unit), origin)


def forcing_table(forcings) -> dict:
    """{name: forcing} of the Scene's *forcings* (a mapping, or a DataFrame indexed by time, one column each)."""
    if forcings is None:
        return {}
    if isinstance(forcings, Mapping):
        return dict(forcings)
    if hasattr(forcings, "columns"):
        return {str(column): forcings[column] for column in forcings.columns}
    raise TypeError(f"forcings: a dict {{name: forcing}} or a DataFrame indexed by time, got {type(forcings).__name__}")
