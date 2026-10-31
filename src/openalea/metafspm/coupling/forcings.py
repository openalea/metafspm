"""
Forcings, the time-varying inputs of a scene (meteo, a measured temperature, a shoot's supply)::

  Scene(..., forcings={"PARi": series, "soil_temperature": 15., "struct_mass_input": function})
  Scene(..., forcings=table)                  a DataFrame indexed by time (s), one column per forcing

A forcing is a number; a callable of the scene time (s); a pandas Series indexed by time (s); or a (times, values)
pair. Tables are linearly interpolated (and held constant beyond their ends). A component declaring the name with
forcing_variable reads it with self.forcing(name); otherwise the Scene writes it, at the start of each step, into
the input or parameter of that name of every component declaring it (not provided by a coupling).

  as_forcing(value, time_unit=86400.)         a user's input in days (a number, a function of the day, a Series
                                              indexed by day, one value per day from day 0) as a forcing
  forcing_value(forcing, time)                its value at *time* (s)
"""
from typing import Mapping

import numpy as np


def _series(value) -> bool:
    return hasattr(value, "index") and hasattr(value, "to_numpy")


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

    def __init__(self, value, time_unit: float, origin: float):
        self.value, self.time_unit, self.origin = value, float(time_unit), float(origin)

    def __call__(self, time: float) -> float:
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


def as_forcing(value, time_unit: float = 1., origin: float = 0.):
    """
    A forcing of the scene time from a user's input given in *time_unit* (s; 86400 for days): a number, a function
    of that unit, a pandas Series indexed by it, or a sequence (one value per unit from 0), linearly interpolated.
    *origin*: the input's time at the scene's time 0 (e.g. the hour of the day the scene starts).
    """
    if np.ndim(value) == 0 and not callable(value) and not _series(value):
        return float(value)
    return _InUnit(value, time_unit, origin)


def forcing_table(forcings) -> dict:
    """{name: forcing} of the Scene's *forcings* (a mapping, or a DataFrame indexed by time, one column each)."""
    if forcings is None:
        return {}
    if isinstance(forcings, Mapping):
        return dict(forcings)
    if hasattr(forcings, "columns"):
        return {str(column): forcings[column] for column in forcings.columns}
    raise TypeError(f"forcings: a dict {{name: forcing}} or a DataFrame indexed by time, got {type(forcings).__name__}")
