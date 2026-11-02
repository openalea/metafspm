"""
Forcings, the time-varying inputs of a scene (meteo, a measured temperature, a shoot's supply)::

  Scene(..., forcings={"PARi": series, "soil_temperature": 15., "struct_mass_input": function})
  Scene(..., forcings=table)                  a DataFrame indexed by time (s), one column per forcing

A forcing is a number; a callable of the scene time (s); a pandas Series indexed by time (s); or a (times, values)
pair. Tables are linearly interpolated (and held constant beyond their ends). A callable of the time and the entity
ids, (time, ids) -> one value per entity, varies in space (a soil temperature by depth, an input per segment): the
Scene writes it into the input or parameter of that name, on the entities of its location. A component declaring
the name with forcing_variable reads it with self.forcing(name); otherwise the Scene writes it, at the start of each step, into
the input or parameter of that name of every component declaring it (not provided by a coupling)::

  as_forcing(value, time_unit=86400.)         a user's input in days (a number, a function of the day, a Series
                                              indexed by day, one value per day from day 0) as a forcing
  forcing_value(forcing, time)                its value at *time* (s)
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

    def __call__(self, time: float) -> float:
        if self.origin == "calendar":
            raise ValueError("a forcing of origin 'calendar' needs the Scene's start=(day_of_year, hour)")
        when = self.origin + time / self.time_unit
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
